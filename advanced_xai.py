import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lime import lime_image
from pytorch_grad_cam import GradCAM
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import cv2
import logging
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import os
import warnings
import time
import gc
warnings.filterwarnings('ignore')

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: 'shap' not installed. Install with: pip install shap")

from scipy import stats

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"xai_covid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from config import Config
    DEFAULT_DEVICE = Config.DEVICE
    CLASSES = Config.CLASSES
    NUM_CLASSES = Config.NUM_CLASSES
    IMAGE_SIZE = Config.IMAGE_SIZE
except:
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASSES = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
    NUM_CLASSES = 4
    IMAGE_SIZE = 224

MODEL_PATH = 'outputs/models/best_model.pth'
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def load_model(model_path=MODEL_PATH, num_classes=NUM_CLASSES, device=DEFAULT_DEVICE):
    try:
        from models import ImprovedTransformerEnsembleModel
    except:
        logger.error("Could not import ImprovedTransformerEnsembleModel")
        raise

    if str(device) == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    new_sd = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        name = name.replace('vit_base.', 'vit.')
        name = name.replace('swin_base.', 'swin.')
        name = name.replace('convnext_base.', 'convnext.')
        name = name.replace('ensemble_layer.', 'feature_fusion.')
        new_sd[name] = v

    model = ImprovedTransformerEnsembleModel(
        num_classes=num_classes, use_aux_loss=True, pretrained=False
    ).to(device)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    clf_keys = ['head', 'fc', 'classifier', 'final']
    missing    = [k for k in missing    if not any(c in k for c in clf_keys)]
    unexpected = [k for k in unexpected if not any(c in k for c in clf_keys)]
    if missing    and len(missing)    < 20: logger.info(f"Missing {len(missing)} keys")
    if unexpected and len(unexpected) < 20: logger.info(f"Unexpected {len(unexpected)} keys")
    model.eval()
    logger.info("Model loaded successfully")
    return model


def find_last_conv2d(model: nn.Module) -> Tuple[str, nn.Module]:
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = (name, mod)
    if last is None:
        raise ValueError("No nn.Conv2d found in model")
    logger.info(f"CAM target layer: {last[0]}")
    return last


class GroundTruthMetrics:
    @staticmethod
    def compute_iou(explanation: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        binary = (explanation >= threshold).astype(np.float32)
        inter  = (binary * gt_mask).sum()
        union  = ((binary + gt_mask) > 0).astype(np.float32).sum()
        return float(inter / (union + 1e-8))

    @staticmethod
    def pointing_game(explanation: np.ndarray, gt_mask: np.ndarray) -> float:
        idx = np.unravel_index(np.argmax(explanation), explanation.shape)
        return float(gt_mask[idx] > 0)

    @staticmethod
    def energy_inside_bbox(explanation: np.ndarray, gt_mask: np.ndarray) -> float:
        return float((explanation * gt_mask).sum() / (explanation.sum() + 1e-8))


class StatisticalTests:
    @staticmethod
    def friedman_nemenyi(results_dict: Dict[str, List[float]], metric_name: str = 'metric') -> Dict:
        methods = list(results_dict.keys())
        n = min(len(v) for v in results_dict.values())
        if n < 3:
            return {'error': 'Need >= 3 samples', 'n': n}

        data = np.array([results_dict[m][:n] for m in methods]).T.astype(np.float64)
        if np.all(np.isnan(data)) or np.all(data == 0):
            return {'metric': metric_name, 'friedman_statistic': np.nan,
                    'friedman_p_value': np.nan, 'n_samples': n,
                    'methods': methods, 'significant': False, 'error': 'All NaN/zero'}

        try:
            stat, p = stats.friedmanchisquare(*[data[:, i] for i in range(len(methods))])
        except Exception as e:
            return {'metric': metric_name, 'friedman_statistic': np.nan,
                    'friedman_p_value': np.nan, 'n_samples': n,
                    'methods': methods, 'significant': False, 'error': str(e)}

        result = {'metric': metric_name, 'friedman_statistic': float(stat),
                  'friedman_p_value': float(p), 'n_samples': n,
                  'methods': methods, 'significant': bool(p < 0.05)}

        if p < 0.05 and HAS_POSTHOCS:
            try:
                nem = sp.posthoc_nemenyi_friedman(data)
                nem.index = methods; nem.columns = methods
                result['nemenyi_p_values'] = nem.to_dict()
                pairs = []
                for i, m1 in enumerate(methods):
                    for j, m2 in enumerate(methods):
                        if i < j and nem.iloc[i, j] < 0.05:
                            pairs.append((m1, m2, float(nem.iloc[i, j])))
                result['significant_pairs'] = pairs
            except Exception as e:
                result['nemenyi_error'] = str(e)
        return result

    @staticmethod
    def run_all_tests(all_results: Dict[str, Dict[str, List[float]]]) -> Dict:
        out = {}
        for metric in ['insertion_auc', 'deletion_auc', 'robustness_ssim', 'robustness_rank_corr']:
            data = {}
            for method, vals in all_results.items():
                if metric in vals and vals[metric]:
                    valid = [v for v in vals[metric] if not np.isnan(v)]
                    if valid:
                        data[method] = valid
            if len(data) >= 2:
                out[metric] = StatisticalTests.friedman_nemenyi(data, metric)
        return out


class PyramidXAIFusion:
    def normalize(self, arr):
        arr = arr.copy().astype(np.float64)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo)

    def pyramid_fusion(self, gradcam, shap, lime, weights=None):
        if weights is None:
            weights = [0.35, 0.30, 0.25, 0.10]
        scales = [14, 28, 56, 112]
        fused = []
        for scale, sw in zip(scales, weights):
            gs = cv2.resize(self.normalize(gradcam).astype(np.float32), (scale, scale))
            ss = cv2.resize(self.normalize(shap).astype(np.float32),    (scale, scale))
            ls = cv2.resize(self.normalize(lime).astype(np.float32),    (scale, scale))
            f  = 0.5 * gs + 0.3 * ss + 0.2 * ls
            fused.append(sw * cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE)))
        return self.normalize(np.sum(fused, axis=0))

    def attention_fusion(self, gradcam, shap, lime, temperature=0.8):
        g, s, l = self.normalize(gradcam), self.normalize(shap), self.normalize(lime)
        gs = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0.5)
        ss = cv2.GaussianBlur(s.astype(np.float32), (3, 3), 0.5)
        ls = cv2.GaussianBlur(l.astype(np.float32), (3, 3), 0.5)
        stack = np.stack([gs, ss, ls], axis=0)
        exp_s = np.exp((stack - np.max(stack, axis=0, keepdims=True)) / temperature)
        attn  = exp_s / (np.sum(exp_s, axis=0, keepdims=True) + 1e-8)
        orig  = np.stack([g, s, l], axis=0)
        result = cv2.GaussianBlur(np.sum(attn * orig, axis=0).astype(np.float32), (3, 3), 0.3)
        return self.normalize(result)


class EnhancedXAIExplainer:

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or DEFAULT_DEVICE
        self.model.eval()
        self.pyramid_fusion     = PyramidXAIFusion()
        self.background_samples = None
        self.shap_explainer     = None

        self._conv2d_name, self._conv2d_layer = find_last_conv2d(model)

        self.mean_t = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.std_t  = torch.tensor(STD,  dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        logger.info(f"EnhancedXAIExplainer | device={self.device} | CAM->{self._conv2d_name}")

    def _model_forward(self, x):
        if hasattr(self.model, '_original_forward'):
            out = self.model._original_forward(x)
        else:
            out = self.model.forward(x)
        return out['ensemble'] if isinstance(out, dict) else out

    def _create_background_samples(self, train_loader, num_samples=50):
        if self.background_samples is not None:
            return self.background_samples
        bufs = []
        for imgs, _ in train_loader:
            bufs.append(imgs)
            if sum(b.size(0) for b in bufs) >= num_samples:
                break
        self.background_samples = torch.cat(bufs, dim=0)[:num_samples].to(self.device)
        logger.info(f"Background samples: {self.background_samples.shape}")
        return self.background_samples

    def _initialize_shap_explainer(self, background_samples):
        if not HAS_SHAP or self.shap_explainer is not None:
            return
        try:
            bg = background_samples[:min(50, len(background_samples))].to(self.device).float()

            class ModelWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    out = self.m(x)
                    return out['ensemble'] if isinstance(out, dict) else out

            self.shap_explainer = shap.DeepExplainer(ModelWrapper(self.model).to(self.device), bg)
            logger.info(f"SHAP DeepExplainer initialized with {len(bg)} background samples")
        except Exception as e:
            logger.error(f"SHAP init failed: {e}")
            self.shap_explainer = None

    def get_gradcam_explanations(self, images) -> List[np.ndarray]:
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        self.model._original_forward = self.model.forward
        self.model.forward = self._model_forward
        try:
            cam_obj = GradCAM(model=self.model, target_layers=[self._conv2d_layer], reshape_transform=None)
            cams    = cam_obj(images, targets=None)
            results = []
            for c in cams:
                if c.shape != (IMAGE_SIZE, IMAGE_SIZE):
                    c = cv2.resize(c, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                results.append(c.astype(np.float32))
            return results
        finally:
            self.model.forward = self.model._original_forward
            delattr(self.model, '_original_forward')

    def get_gradcam_plus_plus(self, images) -> np.ndarray:
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        self.model.eval()

        _acts, _grads = [], []

        def fwd(m, i, o): _acts.append(o.detach().clone())
        def bwd(m, gi, go):
            if go[0] is not None: _grads.append(go[0].detach().clone())

        h1 = self._conv2d_layer.register_forward_hook(fwd)
        h2 = self._conv2d_layer.register_full_backward_hook(bwd)
        try:
            self.model.zero_grad(set_to_none=True)
            out = self._model_forward(images)
            tc  = out.argmax(dim=1).item()
            out[0, tc].backward()

            if not _acts or not _grads:
                return self.get_gradcam_explanations(images)[0]

            act  = _acts[0]
            grad = _grads[0]
            grad_2  = grad ** 2; grad_3 = grad ** 3
            sum_act = act.sum(dim=(2, 3), keepdim=True)
            alpha   = grad_2 / (2.0 * grad_2 + sum_act * grad_3 + 1e-8)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam     = F.relu((weights * act).sum(dim=1)).squeeze().cpu().numpy()

            if cam.max() < 1e-8:
                return self.get_gradcam_explanations(images)[0]

            cam = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
            lo, hi = cam.min(), cam.max()
            return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)
        except Exception as e:
            logger.error(f"GradCAM++ error: {e}")
            return self.get_gradcam_explanations(images)[0]
        finally:
            h1.remove(); h2.remove()
            self.model.zero_grad(set_to_none=True)

    def get_integrated_gradients(self, images, steps: int = 64) -> np.ndarray:
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device).float()
        self.model.eval()

        if self.background_samples is not None:
            baseline = self.background_samples.mean(dim=0, keepdim=True).to(self.device).float()
        else:
            baseline = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1).expand_as(images).clone().float()

        with torch.no_grad():
            out = self._model_forward(images)
            tc  = out.argmax(dim=1).item()

        delta    = (images - baseline).detach()
        grad_acc = torch.zeros_like(images)
        alphas   = torch.linspace(1.0 / steps, 1.0, steps, device=self.device)

        for alpha in alphas:
            interp = (baseline + alpha * delta).float().detach().requires_grad_(True)
            out    = self._model_forward(interp)
            self.model.zero_grad(set_to_none=True)
            out[0, tc].backward()
            if interp.grad is not None:
                grad_acc += interp.grad.detach()

        saliency = ((grad_acc / steps).detach() * delta).abs().sum(dim=1).squeeze().cpu().numpy()
        if saliency.shape != (IMAGE_SIZE, IMAGE_SIZE):
            saliency = cv2.resize(saliency.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        saliency_u8 = (saliency / (saliency.max() + 1e-8) * 255).astype(np.uint8)
        saliency    = cv2.bilateralFilter(saliency_u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)
        p95 = np.percentile(saliency, 95)
        if p95 > 1e-8:
            saliency = np.clip(saliency, 0, p95)
        lo, hi = saliency.min(), saliency.max()
        if hi - lo < 1e-8:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return ((saliency - lo) / (hi - lo)).astype(np.float32)

    def get_rise_explanations(self, images, num_masks: int = 6000, p: float = 0.5,
                              mask_size: int = 8, batch_size: int = 64) -> np.ndarray:
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        self.model.eval()

        with torch.no_grad():
            out      = self._model_forward(images)
            tc       = out.argmax(dim=1).item()
            base_prob = F.softmax(out, dim=1)[0, tc].item()

        saliency = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        density  = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

        for bs_start in range(0, num_masks, batch_size):
            bs  = min(batch_size, num_masks - bs_start)
            h = w = mask_size + 1
            low     = (torch.rand(bs, 1, h, w, device=self.device) < p).float()
            shift_y = torch.randint(0, h - mask_size + 1, (bs,)).tolist()
            shift_x = torch.randint(0, w - mask_size + 1, (bs,)).tolist()
            cropped = torch.stack([
                low[i, :, shift_y[i]:shift_y[i]+mask_size, shift_x[i]:shift_x[i]+mask_size]
                for i in range(bs)
            ])
            masks  = F.interpolate(cropped, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
            masked = images * masks
            with torch.no_grad():
                probs = F.softmax(self._model_forward(masked), dim=1)[:, tc]
            masks_np = masks[:, 0].cpu().numpy()
            probs_np = probs.cpu().numpy()
            for i in range(bs):
                saliency += probs_np[i] * masks_np[i]
                density  += masks_np[i]

        saliency = np.where(density > 0, saliency / (density + 1e-8), 0.0).astype(np.float32)
        saliency = np.clip(saliency - base_prob * p, 0.0, None).astype(np.float32)
        saliency = cv2.GaussianBlur(saliency, (9, 9), 2.0)
        p30 = np.percentile(saliency[saliency > 0], 30) if (saliency > 0).any() else 0
        saliency = np.where(saliency > p30, saliency - p30, 0.0).astype(np.float32)
        lo, hi = saliency.min(), saliency.max()
        if hi - lo < 1e-8:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return ((saliency - lo) / (hi - lo)).astype(np.float32)

    def get_shap_explanations(self, images, **kwargs) -> np.ndarray:
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        bg = self.background_samples if self.background_samples is not None else torch.zeros_like(images)
        self.model.eval()
        with torch.no_grad():
            out = self._model_forward(images)
            pc  = out.argmax(dim=1).item()

        n_refs  = min(15, bg.size(0))
        refs    = bg[torch.randperm(bg.size(0))[:n_refs]]
        n_steps = 20
        grad_acc = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)

        for i in range(n_refs):
            ref   = refs[i:i+1]
            delta = images - ref
            for step in range(1, n_steps + 1):
                interp = (ref + (step / n_steps) * delta).float().detach().requires_grad_(True)
                o = self._model_forward(interp)
                self.model.zero_grad()
                o[0, pc].backward()
                if interp.grad is not None:
                    grad_acc += interp.grad.detach()

        avg_grad = grad_acc / (n_refs * n_steps)
        diff = (images - refs.mean(dim=0, keepdim=True)).detach()
        attr = (avg_grad * diff).abs().sum(dim=1).squeeze().detach().cpu().numpy()
        attr = cv2.GaussianBlur(attr.astype(np.float32), (9, 9), 3.0)
        lo, hi = attr.min(), attr.max()
        return ((attr - lo) / (hi - lo + 1e-8)).astype(np.float32)

    def get_lime_explanations(self, image, num_samples=300, num_features=8) -> np.ndarray:
        if len(image.shape) == 4:
            image = image[0]
        img_np      = image.detach().cpu().numpy().transpose(1, 2, 0)
        img_display = np.clip(img_np * STD + MEAN, 0, 1)

        def predict_fn(batch):
            t = torch.tensor(batch, dtype=torch.float32).permute(0, 3, 1, 2)
            t = ((t - self.mean_t.cpu()) / self.std_t.cpu()).to(self.device)
            self.model.eval()
            with torch.no_grad():
                out = self._model_forward(t)
                return F.softmax(out, dim=1).cpu().numpy()

        def seg_fn(img):
            return slic(img, n_segments=50, compactness=5, sigma=2, start_label=1)

        explainer   = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_display, predict_fn, top_labels=1, hide_color=None,
            num_samples=num_samples, num_features=num_features,
            random_seed=42, segmentation_fn=seg_fn
        )
        pred_class  = explanation.top_labels[0]
        local_exp   = explanation.local_exp[pred_class]
        segments    = seg_fn(img_display)
        heatmap     = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for seg_id, weight in local_exp:
            heatmap[segments == seg_id] = weight
        heatmap = np.abs(heatmap)
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 3.0)
        lo, hi  = heatmap.min(), heatmap.max()
        return ((heatmap - lo) / (hi - lo + 1e-8)).astype(np.float32)

    def faithfulness_insertion_deletion(self, image, explanation, num_steps=100):
        image    = image.to(self.device)
        baseline = torch.zeros_like(image).to(self.device)

        with torch.no_grad():
            out        = self._model_forward(image)
            orig_class = F.softmax(out, dim=1).argmax(dim=1).item()

        flat      = explanation.flatten()
        order     = np.argsort(flat)[::-1]
        total     = len(order)
        step_size = max(1, total // num_steps)

        batch_ins, batch_del = [], []
        for step in range(num_steps + 1):
            n_px = min(step * step_size, total)
            mask = np.zeros_like(flat)
            if n_px > 0: mask[order[:n_px]] = 1.0
            mt   = torch.from_numpy(mask.reshape(IMAGE_SIZE, IMAGE_SIZE)).float().to(self.device).unsqueeze(0).unsqueeze(0)
            batch_ins.append(baseline * (1 - mt) + image * mt)
            batch_del.append(image    * (1 - mt) + baseline * mt)

        batch_ins = torch.cat(batch_ins, dim=0)
        batch_del = torch.cat(batch_del, dim=0)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(str(self.device) == 'cuda')):
            ins_scores = F.softmax(self._model_forward(batch_ins).float(), dim=1)[:, orig_class].cpu().numpy()
            del_scores = F.softmax(self._model_forward(batch_del).float(), dim=1)[:, orig_class].cpu().numpy()

        return {
            'insertion_auc': float(np.trapz(ins_scores, dx=1.0 / num_steps)),
            'deletion_auc':  float(np.trapz(del_scores, dx=1.0 / num_steps))
        }

    def robustness_spearman(self, original_map: np.ndarray, perturbed_map: np.ndarray) -> float:
        corr, _ = stats.spearmanr(original_map.flatten(), perturbed_map.flatten())
        return 0.0 if np.isnan(corr) else float(np.clip(corr, 0, 1))

    def robustness_ssim_per_method(self, image, explanation_map, method_name,
                                    num_perturbations=10, noise_std=0.3, shap_map=None) -> Dict[str, float]:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        ssim_scores, rank_scores = [], []
        for _ in range(num_perturbations):
            noise     = torch.randn_like(image) * noise_std
            perturbed = torch.clamp(image + noise, 0, 1)
            try:
                if method_name == 'GradCAM':
                    p_exp = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_explanations(perturbed)[0]))
                elif method_name == 'GradCAM++':
                    p_exp = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_plus_plus(perturbed)))
                elif method_name == 'IntegratedGradients':
                    p_exp = self.pyramid_fusion.normalize(self._ensure_size(self.get_integrated_gradients(perturbed)))
                elif method_name == 'RISE':
                    p_exp = self.pyramid_fusion.normalize(self._ensure_size(self.get_rise_explanations(perturbed, num_masks=2000)))
                elif method_name == 'LIME':
                    p_exp = self.pyramid_fusion.normalize(self._ensure_size(self.get_lime_explanations(perturbed)))
                elif method_name in ('Pyramid-Fusion', 'Attention-Fusion'):
                    if shap_map is None:
                        logger.warning(f"Robustness skip {method_name}: missing SHAP map"); continue
                    gc = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_explanations(perturbed)[0]))
                    li = self.pyramid_fusion.normalize(self._ensure_size(self.get_lime_explanations(perturbed)))
                    sh = self.pyramid_fusion.normalize(self._ensure_size(shap_map))
                    p_exp = (self.pyramid_fusion.pyramid_fusion if method_name == 'Pyramid-Fusion'
                             else self.pyramid_fusion.attention_fusion)(gc, sh, li)
                else:
                    continue
                ssim_scores.append(ssim(explanation_map, p_exp, data_range=1.0))
                rank_scores.append(self.robustness_spearman(explanation_map, p_exp))
            except Exception as e:
                logger.warning(f"Robustness error {method_name}: {e}"); continue

        return {
            'robustness_ssim':      float(np.clip(np.mean(ssim_scores), 0, 1)) if ssim_scores else 0.0,
            'robustness_rank_corr': float(np.clip(np.mean(rank_scores), 0, 1)) if rank_scores else 0.0
        }

    def hyperparameter_sensitivity(self, explanation):
        results = {"temperature_sweep": {}, "weight_sweep": {}}
        gc  = self.pyramid_fusion.normalize(explanation['gradcam'])
        sh  = self.pyramid_fusion.normalize(self.get_shap_explanations(explanation['image']))
        li  = self.pyramid_fusion.normalize(explanation['lime'])
        for temp in [0.6, 0.8, 1.0]:
            fused = self.pyramid_fusion.attention_fusion(gc, sh, li, temperature=temp)
            fm    = self.faithfulness_insertion_deletion(explanation['image'], fused)
            results["temperature_sweep"][f"temp_{temp}"] = fm['insertion_auc']
        for idx, w in enumerate([[0.35, 0.30, 0.25, 0.10], [0.30, 0.30, 0.30, 0.10], [0.40, 0.30, 0.20, 0.10]]):
            fused = self.pyramid_fusion.pyramid_fusion(gc, sh, li, weights=w)
            fm    = self.faithfulness_insertion_deletion(explanation['image'], fused)
            results["weight_sweep"][f"config_{idx}"] = fm['insertion_auc']
        os.makedirs('visualizations', exist_ok=True)
        with open('visualizations/hyperparameter_sensitivity.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def visualize_all_methods(self, image, save_dir='visualizations', sample_id=0,
                               class_names=None, shap_map=None):
        os.makedirs(save_dir, exist_ok=True)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            out       = self._model_forward(image)
            probs     = F.softmax(out, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        img_np = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)

        gradcam    = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_explanations(image)[0])).astype(np.float32)
        gradcam_pp = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_plus_plus(image))).astype(np.float32)
        ig         = self.pyramid_fusion.normalize(self._ensure_size(self.get_integrated_gradients(image))).astype(np.float32)
        rise       = self.pyramid_fusion.normalize(self._ensure_size(self.get_rise_explanations(image))).astype(np.float32)
        lime_map   = self.pyramid_fusion.normalize(self._ensure_size(self.get_lime_explanations(image))).astype(np.float32)
        shap_map   = self.pyramid_fusion.normalize(self._ensure_size(
            shap_map if shap_map is not None else self.get_shap_explanations(image)
        )).astype(np.float32)
        pyramid    = self.pyramid_fusion.pyramid_fusion(gradcam, shap_map, lime_map).astype(np.float32)
        attention  = self.pyramid_fusion.attention_fusion(gradcam, shap_map, lime_map).astype(np.float32)

        cn  = class_names[pred_class] if (class_names and pred_class < len(class_names)) else f"Class {pred_class}"
        fig = plt.figure(figsize=(28, 10))
        gs_fig = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
        fig.suptitle(f'XAI - Sample {sample_id} | Pred: {cn} ({confidence:.1%})', fontsize=16, fontweight='bold')

        for c, (hm, lbl) in enumerate([(None, 'Original'), (gradcam, 'GradCAM'),
                (gradcam_pp, 'GradCAM++'), (ig, 'IntegratedGradients'), (rise, 'RISE')]):
            ax = fig.add_subplot(gs_fig[0, c])
            ax.imshow(img_np)
            if hm is not None: ax.imshow(hm, cmap='jet', alpha=0.45, vmin=0, vmax=1)
            ax.set_title(lbl, fontsize=12, fontweight='bold'); ax.axis('off')

        for c, (hm, lbl) in enumerate([(lime_map, 'LIME'),
                (pyramid, 'Pyramid Fusion'), (attention, 'Attention Fusion')]):
            ax = fig.add_subplot(gs_fig[1, c])
            ax.imshow(img_np)
            ax.imshow(hm.astype(np.float32), cmap='jet', alpha=0.45, vmin=0, vmax=1)
            ax.set_title(lbl, fontsize=12, fontweight='bold'); ax.axis('off')

        path = os.path.join(save_dir, f'comprehensive_sample_{sample_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        logger.info(f"Saved: {path}")
        return path

    def visualize_comparison_grid(self, images, save_dir='visualizations', class_names=None, max_samples=9):
        os.makedirs(save_dir, exist_ok=True)
        n    = min(len(images), max_samples)
        fig, axes = plt.subplots(n, 7, figsize=(28, 4 * n))
        if n == 1: axes = axes.reshape(1, -1)

        for idx in range(n):
            img = images[idx].unsqueeze(0).to(self.device) if len(images[idx].shape) == 3 else images[idx].to(self.device)
            with torch.no_grad():
                pc = self._model_forward(img).argmax(dim=1).item()
            img_np = np.clip(img[0].detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)

            gc_  = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_explanations(img)[0])).astype(np.float32)
            gcpp = self.pyramid_fusion.normalize(self._ensure_size(self.get_gradcam_plus_plus(img))).astype(np.float32)
            ig_  = self.pyramid_fusion.normalize(self._ensure_size(self.get_integrated_gradients(img))).astype(np.float32)
            rise_= self.pyramid_fusion.normalize(self._ensure_size(self.get_rise_explanations(img))).astype(np.float32)
            li_  = self.pyramid_fusion.normalize(self._ensure_size(self.get_lime_explanations(img))).astype(np.float32)
            sh_  = self.pyramid_fusion.normalize(self._ensure_size(self.get_shap_explanations(img))).astype(np.float32)
            pyr  = self.pyramid_fusion.pyramid_fusion(gc_, sh_, li_).astype(np.float32)
            cn   = class_names[pc] if (class_names and pc < len(class_names)) else f"Class {pc}"

            for col, (ov, title) in enumerate([
                (None, cn), (gc_, 'GradCAM'), (gcpp, 'GradCAM++'),
                (ig_, 'IG'), (rise_, 'RISE'), (li_, 'LIME'), (pyr, 'Pyramid')
            ]):
                axes[idx, col].imshow(img_np)
                if ov is not None: axes[idx, col].imshow(ov, cmap='jet', alpha=0.45, vmin=0, vmax=1)
                axes[idx, col].set_title(title, fontsize=10); axes[idx, col].axis('off')

        plt.tight_layout()
        path = os.path.join(save_dir, 'comparison_grid.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        return path

    @staticmethod
    def _ensure_size(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
            arr = cv2.resize(arr.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        return arr.astype(np.float32)

    def evaluate_xai_with_pyramid_fusion(
        self, test_loader, num_samples=300,
        train_loader=None, batch_size=16,
        visualize_samples=True, vis_save_dir='visualizations',
        class_names=None
    ):
        logger.info("=" * 60)
        logger.info("COVID-19 XAI EVALUATION")
        logger.info(f"Classes: {CLASSES}")
        logger.info(f"CAM target: {self._conv2d_name}")
        logger.info("=" * 60)

        start_time = time.time()

        method_configs = {
            'GradCAM': 'gradcam', 'GradCAM++': 'gradcam_pp',
            'IntegratedGradients': 'ig', 'RISE': 'rise',
            'LIME': 'lime',
            'Pyramid-Fusion': 'pyramid', 'Attention-Fusion': 'attention'
        }

        results = {m: {
            'insertion_auc': [], 'deletion_auc': [],
            'robustness_ssim': [], 'robustness_rank_corr': []
        } for m in method_configs}

        if train_loader is not None:
            bg = self._create_background_samples(train_loader, num_samples=50)
            if HAS_SHAP:
                self._initialize_shap_explainer(bg)

        all_explanations = []
        vis_images       = []
        collected        = 0

        logger.info("Collecting explanations...")
        collect_start = time.time()

        for _, (images, _) in enumerate(test_loader):
            if collected >= num_samples: break
            for i in range(images.size(0)):
                if collected >= num_samples: break
                image = images[i:i+1].clone().to(self.device)

                with torch.no_grad():
                    out  = self._model_forward(image)
                    conf = F.softmax(out, dim=1).max(dim=1)[0].item()
                if conf < 0.5: continue

                if visualize_samples and len(vis_images) < 9:
                    vis_images.append(image[0].detach().clone())

                try:
                    gradcam    = self._ensure_size(self.get_gradcam_explanations(image)[0])
                    gradcam_pp = self._ensure_size(self.get_gradcam_plus_plus(image))
                    ig         = self._ensure_size(self.get_integrated_gradients(image))
                    rise       = self._ensure_size(self.get_rise_explanations(image))
                    lime_map   = self._ensure_size(self.get_lime_explanations(image))

                    gradcam, gradcam_pp, ig, rise, lime_map = [
                        self.pyramid_fusion.normalize(x).astype(np.float32)
                        for x in [gradcam, gradcam_pp, ig, rise, lime_map]
                    ]

                    shap_map = self.pyramid_fusion.normalize(
                        self._ensure_size(self.get_shap_explanations(image))
                    ).astype(np.float32)

                    exp_data = {
                        'image':     image.detach().clone(),
                        'gradcam':   gradcam.copy(),   'gradcam_pp': gradcam_pp.copy(),
                        'ig':        ig.copy(),         'rise':       rise.copy(),
                        'lime':      lime_map.copy(),
                        'pyramid':   self.pyramid_fusion.pyramid_fusion(gradcam, shap_map, lime_map).astype(np.float32),
                        'attention': self.pyramid_fusion.attention_fusion(gradcam, shap_map, lime_map).astype(np.float32),
                    }
                    all_explanations.append(exp_data)
                    collected += 1

                    if collected % 10 == 0:
                        elapsed  = time.time() - collect_start
                        avg_time = elapsed / collected
                        logger.info(f"  {collected}/{num_samples} | Avg: {avg_time:.1f}s | ETA: {avg_time*(num_samples-collected)/60:.1f}min")

                    if visualize_samples and collected <= 3:
                        self.visualize_all_methods(
                            image, save_dir=vis_save_dir, sample_id=collected,
                            class_names=class_names, shap_map=shap_map
                        )
                except Exception as e:
                    logger.error(f"Error sample {collected}: {e}")
                    import traceback; logger.error(traceback.format_exc())
                    collected += 1; continue

        if str(self.device) == 'cuda':
            torch.cuda.empty_cache(); gc.collect()

        if visualize_samples and vis_images:
            self.visualize_comparison_grid(vis_images, save_dir=vis_save_dir, class_names=class_names)

        logger.info(f"Total explanations: {len(all_explanations)}")
        if not all_explanations:
            return {'error': 'No explanations collected', 'samples_processed': 0}

        if all_explanations:
            self.hyperparameter_sensitivity(all_explanations[0])

        logger.info("Evaluating faithfulness and robustness...")
        eval_start = time.time()

        for method_name, key in method_configs.items():
            logger.info(f"Evaluating {method_name}...")
            method_start = time.time()
            for idx, ed in enumerate(all_explanations):
                try:
                    fm = self.faithfulness_insertion_deletion(ed['image'], ed[key])
                    results[method_name]['insertion_auc'].append(fm['insertion_auc'])
                    results[method_name]['deletion_auc'].append(fm['deletion_auc'])
                    rob = self.robustness_ssim_per_method(
                        ed['image'], ed[key], method_name,
                        num_perturbations=10, noise_std=0.3, shap_map=ed.get('shap')
                    )
                    results[method_name]['robustness_ssim'].append(rob['robustness_ssim'])
                    results[method_name]['robustness_rank_corr'].append(rob['robustness_rank_corr'])
                    if (idx + 1) % 20 == 0:
                        elapsed = time.time() - method_start
                        logger.info(f"  {idx+1}/{len(all_explanations)} | ETA: {elapsed/(idx+1)*(len(all_explanations)-idx-1)/60:.1f}min")
                except Exception as e:
                    logger.error(f"Eval error {method_name} sample {idx}: {e}")
            logger.info(f"  {method_name} done in {(time.time()-method_start)/60:.1f}min")

        if str(self.device) == 'cuda':
            torch.cuda.empty_cache(); gc.collect()

        total_time   = time.time() - start_time
        stat_results = StatisticalTests.run_all_tests(results) if len(all_explanations) >= 3 else None

        final = {}
        best_insertion, best_method = -1, None

        for method_name, metrics in results.items():
            if not metrics['insertion_auc']: continue
            entry = {}
            for k in ['insertion_auc', 'deletion_auc', 'robustness_ssim', 'robustness_rank_corr']:
                if metrics[k]:
                    entry[k]          = float(np.mean(metrics[k]))
                    entry[f'{k}_std'] = float(np.std(metrics[k]))
            final[method_name] = entry
            if entry.get('insertion_auc', 0) > best_insertion:
                best_insertion = entry.get('insertion_auc', 0)
                best_method    = method_name

        final['best_method']         = best_method
        final['samples_processed']   = len(all_explanations)
        final['statistical_tests']   = stat_results
        final['cam_target_layer']    = self._conv2d_name
        final['total_time_minutes']  = total_time / 60
        final['avg_time_per_sample'] = total_time / len(all_explanations)
        final['classes']             = CLASSES

        self._print_results(final)
        if visualize_samples:
            self._plot_results(final, save_dir=vis_save_dir)
            if stat_results:
                self._plot_statistical_tests(stat_results, save_dir=vis_save_dir)

        results_path = os.path.join(vis_save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self._make_serializable(final), f, indent=2)
        logger.info(f"Results saved: {results_path}")
        return final

    def _make_serializable(self, obj):
        if isinstance(obj, dict):                       return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):              return [self._make_serializable(i) for i in obj]
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.bool_):                  return bool(obj)
        if isinstance(obj, np.ndarray):                return obj.tolist()
        if isinstance(obj, bool):                      return obj
        return obj

    def _print_results(self, final):
        sep = "=" * 120
        print(f"\n{sep}")
        print(f"COVID-19 XAI RESULTS | Classes: {final.get('classes',[])}".center(120))
        print(sep)
        hdr = f"{'Method':<22} {'Insertion ^':<18} {'Deletion':<18} {'SSIM ^':<18} {'RankCorr ^':<18}"
        print(hdr); print("-" * 120)
        for mn in ['GradCAM', 'GradCAM++', 'IntegratedGradients', 'RISE', 'LIME', 'Pyramid-Fusion', 'Attention-Fusion']:
            if mn not in final or not isinstance(final[mn], dict): continue
            m   = final[mn]
            row = f"{mn:<22}"
            for k in ['insertion_auc', 'deletion_auc', 'robustness_ssim', 'robustness_rank_corr']:
                row += f" {m.get(k,0):.4f}±{m.get(f'{k}_std',0):.4f}   "
            print(row)
        print(sep)
        print(f"Best Method: {final.get('best_method','N/A')} | Total: {final.get('total_time_minutes',0):.1f}min | "
              f"CAM: {final.get('cam_target_layer','N/A')}".center(120))
        if final.get('statistical_tests'):
            print("\n" + "STATISTICAL SIGNIFICANCE".center(120))
            for metric, res in final['statistical_tests'].items():
                sig = "SIGNIFICANT (p<0.05)" if res.get('significant') else "Not Significant"
                print(f"\n{metric.upper()}: chi-sq={res.get('friedman_statistic',0):.3f}, p={res.get('friedman_p_value',1):.6f} -> {sig}")
                if res.get('significant_pairs'):
                    for m1, m2, p in res.get('significant_pairs', []):
                        print(f"  {m1} vs {m2}: p={p:.6f}")
        print(sep)

    def _plot_results(self, results, save_dir='visualizations'):
        os.makedirs(save_dir, exist_ok=True)
        methods = ['GradCAM', 'GradCAM++', 'IntegratedGradients', 'RISE', 'LIME', 'Pyramid-Fusion', 'Attention-Fusion']
        metrics = ['insertion_auc', 'deletion_auc', 'robustness_ssim', 'robustness_rank_corr']
        titles  = {
            'insertion_auc':        ('Insertion AUC',   'steelblue'),
            'deletion_auc':         ('Deletion AUC',    'coral'),
            'robustness_ssim':      ('Robustness SSIM', 'forestgreen'),
            'robustness_rank_corr': ('Robustness Rank', 'purple'),
        }
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        x = np.arange(len(methods))
        for ai, metric in enumerate(metrics):
            ax    = axes[ai]
            means = [results.get(m, {}).get(metric, 0) if isinstance(results.get(m), dict) else 0 for m in methods]
            stds  = [results.get(m, {}).get(f'{metric}_std', 0) if isinstance(results.get(m), dict) else 0 for m in methods]
            lbl, clr = titles[metric]
            bars = ax.bar(x, means, 0.6, yerr=stds, capsize=5, alpha=0.8, color=clr, edgecolor='black')
            ax.set_title(lbl, fontsize=13, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
            ax.set_ylim([0, 1.0]); ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'covid_results_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_statistical_tests(self, stat_results, save_dir='visualizations'):
        os.makedirs(save_dir, exist_ok=True)
        items = [(k, v) for k, v in stat_results.items()
                 if 'friedman_statistic' in v and not np.isnan(v.get('friedman_statistic', float('nan')))]
        if not items: return
        fig, axes = plt.subplots(1, len(items), figsize=(9 * len(items), 7))
        if len(items) == 1: axes = [axes]
        for ax, (metric, res) in zip(axes, items):
            methods = res.get('methods', [])
            n = len(methods)
            if 'nemenyi_p_values' in res and n > 0:
                matrix = np.ones((n, n))
                pv = res['nemenyi_p_values']
                for i, m1 in enumerate(methods):
                    for j, m2 in enumerate(methods):
                        if m1 in pv and m2 in pv[m1]:
                            matrix[i, j] = pv[m1][m2]
                im    = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
                short = [m.replace('-Fusion', '').replace('Integrated', 'IG') for m in methods]
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(short, fontsize=8)
                for i in range(n):
                    for j in range(n):
                        clr = 'white' if matrix[i, j] < 0.05 else 'black'
                        ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=7, color=clr)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, f'Friedman p={res["friedman_p_value"]:.4f}',
                        ha='center', va='center', transform=ax.transAxes)
            sig = "p<0.05" if res.get('significant') else "n.s."
            ax.set_title(f'{metric}\nchi-sq={res["friedman_statistic"]:.2f} ({sig})', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'statistical_tests.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    Config.set_seed(Config.SEED)
    device = str(DEFAULT_DEVICE)
    model  = load_model(MODEL_PATH, num_classes=NUM_CLASSES, device=device)

    from data_loader import DataManager
    dm = DataManager()
    train_loader, _, test_loader, _ = dm.get_data_loaders()

    explainer = EnhancedXAIExplainer(model=model, device=device)

    results = explainer.evaluate_xai_with_pyramid_fusion(
        test_loader=test_loader, num_samples=100,
        train_loader=train_loader, batch_size=Config.BATCH_SIZE,
        visualize_samples=True,
        vis_save_dir=Config.VISUALIZATION_DIR,
        class_names=CLASSES
    )
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()
