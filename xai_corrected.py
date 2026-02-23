import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from typing import Optional, Callable
import gc
from scipy import stats

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def find_last_conv2d(model: nn.Module) -> tuple:
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = (name, mod)
    if last is None:
        raise ValueError("No nn.Conv2d found anywhere in model")
    return last


def get_gradcam_explanation(model, image: torch.Tensor,
                            target_layer: nn.Module,
                            target_class: int = None) -> np.ndarray:
    device = next(model.parameters()).device
    image  = image.to(device)
    model.eval()

    _acts, _grads = [], []

    def fwd(m, i, o): _acts.append(o.detach())
    def bwd(m, gi, go): _grads.append(go[0].detach())

    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)

    try:
        out = model(image)
        if isinstance(out, dict): out = out['ensemble']
        if target_class is None: target_class = out.argmax(dim=1).item()

        model.zero_grad()
        out[0, target_class].backward()

        if not _acts or not _grads:
            return np.zeros((224, 224), dtype=np.float32)

        act, grad = _acts[0], _grads[0]
        w   = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)
    finally:
        h1.remove(); h2.remove(); model.zero_grad()


def get_gradcam_plus_plus(model, image: torch.Tensor,
                          target_layer: nn.Module,
                          target_class: int = None) -> np.ndarray:
    device = next(model.parameters()).device
    image  = image.to(device).requires_grad_(True)
    model.eval()

    _acts, _grads = [], []

    def fwd(m, i, o): _acts.append(o.detach())
    def bwd(m, gi, go): _grads.append(go[0].detach())

    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)

    try:
        out = model(image)
        if isinstance(out, dict): out = out['ensemble']
        if target_class is None: target_class = out.argmax(dim=1).item()

        model.zero_grad()
        out[0, target_class].backward()

        if not _acts or not _grads:
            return np.zeros((224, 224), dtype=np.float32)

        act, grad = _acts[0], _grads[0]
        grad_2 = grad.pow(2)
        grad_3 = grad.pow(3)
        sum_act     = act.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2.0 * grad_2 + sum_act * grad_3 + 1e-8
        alpha   = grad_2 / alpha_denom
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)
    finally:
        h1.remove(); h2.remove(); model.zero_grad()


def get_integrated_gradients(model, image: torch.Tensor,
                             target_class: int = None,
                             steps: int = 64,
                             background: torch.Tensor = None) -> np.ndarray:
    device = next(model.parameters()).device
    image  = image.to(device).float()
    model.eval()

    if background is not None:
        baseline = background.mean(dim=0, keepdim=True).to(device).float()
    else:
        mean_val = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        baseline = mean_val.expand_as(image).clone().float()

    with torch.no_grad():
        out = model(image)
        if isinstance(out, dict): out = out['ensemble']
        if target_class is None: target_class = out.argmax(dim=1).item()

    delta    = (image - baseline).detach()
    grad_acc = torch.zeros_like(image)
    alphas   = torch.linspace(1.0 / steps, 1.0, steps, device=device)

    for alpha in alphas:
        interp = (baseline + alpha * delta).float().detach().requires_grad_(True)
        out    = model(interp)
        if isinstance(out, dict): out = out['ensemble']
        model.zero_grad(set_to_none=True)
        out[0, target_class].backward()
        if interp.grad is not None:
            grad_acc += interp.grad.detach()

    avg_grad = (grad_acc / steps).detach()
    saliency = (avg_grad * delta).abs().sum(dim=1).squeeze().detach().cpu().numpy()

    if saliency.shape != (224, 224):
        saliency = cv2.resize(saliency.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)

    saliency_u8 = (saliency / (saliency.max() + 1e-8) * 255).astype(np.uint8)
    saliency = cv2.bilateralFilter(saliency_u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)

    p95 = np.percentile(saliency, 95)
    if p95 > 1e-8:
        saliency = np.clip(saliency, 0, p95)

    lo, hi = saliency.min(), saliency.max()
    if hi - lo < 1e-8:
        return np.zeros((224, 224), dtype=np.float32)

    return ((saliency - lo) / (hi - lo)).astype(np.float32)


def get_rise_explanation(model, image: torch.Tensor,
                         target_class: int = None,
                         num_masks: int = 6000,
                         p: float = 0.25,
                         mask_size: int = 5,
                         batch_size: int = 64) -> np.ndarray:
    device = next(model.parameters()).device
    image  = image.to(device)
    model.eval()

    with torch.no_grad():
        out = model(image)
        if isinstance(out, dict): out = out['ensemble']
        if target_class is None: target_class = out.argmax(dim=1).item()
        base_prob = F.softmax(out, dim=1)[0, target_class].item()

    saliency = np.zeros((224, 224), dtype=np.float64)
    density  = np.zeros((224, 224), dtype=np.float64)

    for batch_start in range(0, num_masks, batch_size):
        bs = min(batch_size, num_masks - batch_start)

        h = w = mask_size + 1
        low     = (torch.rand(bs, 1, h, w, device=device) < p).float()
        shift_y = torch.randint(0, h - mask_size + 1, (bs,)).tolist()
        shift_x = torch.randint(0, w - mask_size + 1, (bs,)).tolist()

        # slicing GPU tensor with int indices keeps result on GPU
        cropped = torch.stack([
            low[i, :, shift_y[i]:shift_y[i] + mask_size, shift_x[i]:shift_x[i] + mask_size]
            for i in range(bs)
        ])  # (bs, 1, mask_size, mask_size) on device

        masks  = F.interpolate(cropped, size=(224, 224), mode='bilinear', align_corners=False)
        masked = image * masks  # both on device

        with torch.no_grad():
            bout = model(masked)
            if isinstance(bout, dict): bout = bout['ensemble']
            probs = F.softmax(bout, dim=1)[:, target_class]

        # move to CPU once per batch
        masks_np = masks[:, 0].cpu().numpy()
        probs_np = probs.cpu().numpy()
        for i in range(bs):
            saliency += probs_np[i] * masks_np[i]
            density  += masks_np[i]

    # Density normalization (key fix vs simple averaging)
    saliency = np.where(density > 0, saliency / (density + 1e-8), 0.0).astype(np.float32)

    # Subtract baseline contribution
    saliency = np.clip(saliency - base_prob * p, 0.0, None).astype(np.float32)
    saliency  = cv2.GaussianBlur(saliency, (9, 9), 2.0)

    # Suppress bottom 30th percentile background noise
    p30 = np.percentile(saliency[saliency > 0], 30) if (saliency > 0).any() else 0
    saliency = np.where(saliency > p30, saliency - p30, 0.0).astype(np.float32)

    lo, hi = saliency.min(), saliency.max()
    if hi - lo < 1e-8:
        return np.zeros((224, 224), dtype=np.float32)

    return ((saliency - lo) / (hi - lo)).astype(np.float32)


def create_shap_background(train_loader, n: int = 30,
                           device: str = 'cuda') -> torch.Tensor:
    bufs = []
    for imgs, _ in train_loader:
        bufs.append(imgs)
        if sum(b.size(0) for b in bufs) >= n:
            break
    return torch.cat(bufs, dim=0)[:n].to(device)


def get_gradient_shap_explanation(model, image: torch.Tensor,
                                   background: torch.Tensor,
                                   target_class: int = None,
                                   n_samples: int = 50,
                                   noise_std: float = 0.1) -> np.ndarray:
    device     = next(model.parameters()).device
    image      = image.to(device)
    background = background.to(device)
    model.eval()

    with torch.no_grad():
        out = model(image)
        if isinstance(out, dict): out = out['ensemble']
        if target_class is None: target_class = out.argmax(dim=1).item()

    n_bg      = min(5, background.size(0))
    idx       = torch.randperm(background.size(0))[:n_bg]
    baselines  = background[idx]
    grad_sum  = torch.zeros_like(image)

    for baseline in baselines:
        baseline = baseline.unsqueeze(0)
        for _ in range(n_samples // n_bg):
            alpha        = torch.rand(1, device=device).item()
            interpolated = baseline + alpha * (image - baseline)
            noise        = torch.randn_like(interpolated) * noise_std
            interpolated = torch.clamp(interpolated + noise, 0, 1).detach().requires_grad_(True)

            out = model(interpolated)
            if isinstance(out, dict): out = out['ensemble']
            model.zero_grad(set_to_none=True)
            out[0, target_class].backward()

            if interpolated.grad is not None:
                grad_sum += interpolated.grad.detach() * (image - baseline)

    avg_grad = grad_sum / n_samples
    heatmap  = avg_grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
    heatmap  = cv2.GaussianBlur(heatmap.astype(np.float32), (9, 9), 3.0)
    lo, hi   = heatmap.min(), heatmap.max()
    if hi - lo < 1e-8:
        return np.zeros((224, 224), dtype=np.float32)

    return ((heatmap - lo) / (hi - lo + 1e-8)).astype(np.float32)


def get_lime_explanation(model, image: torch.Tensor,
                         device: str = 'cuda',
                         num_samples: int = 300,
                         num_features: int = 10,
                         mean_t: torch.Tensor = None,
                         std_t: torch.Tensor = None) -> np.ndarray:
    if len(image.shape) == 4:
        image = image[0]

    img_np      = image.detach().cpu().numpy().transpose(1, 2, 0)
    img_display = np.clip(img_np * STD + MEAN, 0, 1)

    def seg_fn(img):
        return slic(img, n_segments=50, compactness=5, sigma=2, start_label=1)

    segments = seg_fn(img_display)

    if mean_t is None:
        mean_t = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    if std_t is None:
        std_t  = torch.tensor(STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

    def predict_fn(batch_imgs):
        t = torch.tensor(batch_imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        t = (t - mean_t) / std_t
        model.eval()
        with torch.no_grad():
            out = model(t)
            if isinstance(out, dict): out = out['ensemble']
            return F.softmax(out, dim=1).cpu().numpy()

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_display, predict_fn,
        top_labels=1, hide_color=None,
        num_samples=num_samples, num_features=num_features,
        random_seed=42, segmentation_fn=seg_fn
    )

    pred_class = explanation.top_labels[0]
    local_exp  = explanation.local_exp[pred_class]

    heatmap = np.zeros((224, 224), dtype=np.float32)
    for seg_id, weight in local_exp:
        heatmap[segments == seg_id] = weight

    heatmap = np.abs(heatmap)
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 3.0)
    lo, hi  = heatmap.min(), heatmap.max()
    return ((heatmap - lo) / (hi - lo + 1e-8)).astype(np.float32)


def compute_robustness_ssim(image: torch.Tensor,
                            original_explanation: np.ndarray,
                            explanation_func: Callable,
                            num_perturbations: int = 10,
                            noise_std: float = 0.1,
                            device: str = 'cuda') -> float:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image  = image.to(device)
    noises = torch.randn(num_perturbations, *image.shape, device=device) * noise_std
    perturbed = torch.clamp(image + noises, 0, 1)

    scores = []
    for i in range(num_perturbations):
        try:
            p_exp  = explanation_func(perturbed[i:i+1])
            scores.append(ssim(original_explanation, p_exp, data_range=1.0))
        except Exception:
            continue

    return float(np.clip(np.mean(scores), 0, 1)) if scores else 0.0


def compute_robustness_spearman(original_map: np.ndarray,
                                perturbed_map: np.ndarray) -> float:
    corr, _ = stats.spearmanr(original_map.flatten(), perturbed_map.flatten())
    return 0.0 if np.isnan(corr) else float(corr)


def visualize_xai(image: torch.Tensor, gradcam: np.ndarray,
                  shap: np.ndarray, lime: np.ndarray,
                  save_path: str, pred_class: int = None,
                  confidence: float = None, class_name: str = None):
    if len(image.shape) == 4:
        image = image[0]
    img_np      = image.detach().cpu().numpy().transpose(1, 2, 0)
    img_display = np.clip(img_np * STD + MEAN, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    title_str = class_name or (f"Class {pred_class}" if pred_class is not None else "")
    if confidence is not None:
        title_str += f" ({confidence:.1%})"

    axes[0].imshow(img_display)
    axes[0].set_title('Original', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    for ax, (hmap, name) in zip(axes[1:], [(gradcam, 'Grad-CAM'), (shap, 'GradientSHAP'), (lime, 'LIME')]):
        ax.imshow(img_display)
        ax.imshow(hmap, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.axis('off')

    if title_str:
        fig.suptitle(title_str, fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def visualize_xai_full(image: torch.Tensor, gradcam: np.ndarray,
                       gradcam_pp: np.ndarray, ig: np.ndarray,
                       rise: np.ndarray, shap: np.ndarray,
                       lime: np.ndarray, fusion_maps: dict,
                       save_path: str, pred_class: int = None,
                       confidence: float = None, class_name: str = None,
                       gt_mask: np.ndarray = None):
    if len(image.shape) == 4:
        image = image[0]
    img_np      = image.detach().cpu().numpy().transpose(1, 2, 0)
    img_display = np.clip(img_np * STD + MEAN, 0, 1)

    has_gt = gt_mask is not None and gt_mask.max() > 0
    n_rows = 4 if has_gt else 3

    fig = plt.figure(figsize=(24, 4 * n_rows))
    gs  = fig.add_gridspec(n_rows, 5, hspace=0.25, wspace=0.25)

    title_str = class_name or (f"Class {pred_class}" if pred_class is not None else "")
    if confidence is not None:
        title_str += f" ({confidence:.1%})"
    if title_str:
        fig.suptitle(f"XAI Analysis - {title_str}", fontsize=16, fontweight='bold')

    for c, (hm, label) in enumerate([
        (None, 'Original'), (gradcam, 'GradCAM'),
        (gradcam_pp, 'GradCAM++'), (ig, 'IntegratedGradients'), (rise, 'RISE')
    ]):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(img_display)
        if hm is not None:
            ax.imshow(hm, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    for c, (hm, label) in enumerate([
        (shap, 'GradientSHAP'), (lime, 'LIME'),
        (fusion_maps.get('pyramid', np.zeros((224, 224))), 'Pyramid Fusion'),
        (fusion_maps.get('attention', np.zeros((224, 224))), 'Attention Fusion'),
    ]):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(img_display)
        ax.imshow(hm, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    for c, (hm, label) in enumerate([
        (gradcam, 'GradCAM'), (gradcam_pp, 'GradCAM++'),
        (ig, 'IntegratedGradients'), (rise, 'RISE'), (lime, 'LIME')
    ]):
        ax = fig.add_subplot(gs[2, c])
        im = ax.imshow(hm, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'{label} Heatmap', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    if has_gt:
        ax = fig.add_subplot(gs[3, 0])
        ax.imshow(img_display)
        ax.imshow(gt_mask, cmap='Reds', alpha=0.4)
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')

        for c, (mname, exp) in enumerate([
            ('GradCAM', gradcam), ('GradCAM++', gradcam_pp),
            ('IG', ig), ('Pyramid', fusion_maps.get('pyramid', np.zeros((224, 224))))
        ]):
            ax = fig.add_subplot(gs[3, c + 1])
            ax.imshow(img_display)
            ax.imshow(exp, cmap='jet', alpha=0.45, vmin=0, vmax=1)
            contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = cnt.squeeze()
                if len(pts.shape) == 2:
                    ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), 'lime', linewidth=2)
            binary = (exp >= 0.5).astype(np.float32)
            iou    = (binary * gt_mask).sum() / (((binary + gt_mask) > 0).sum() + 1e-8)
            pg     = float(gt_mask[np.unravel_index(np.argmax(exp), exp.shape)] > 0)
            ax.set_title(f'{mname}\nIoU={iou:.3f}  PG={pg:.0f}', fontsize=11)
            ax.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def patch_explainer(explainer, train_loader=None):
    model  = explainer.model
    device = explainer.device

    mean_t = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

    if train_loader is not None:
        explainer._shap_bg = create_shap_background(train_loader, 30, device)
    elif hasattr(explainer, 'background_samples') and explainer.background_samples is not None:
        explainer._shap_bg = explainer.background_samples
    else:
        explainer._shap_bg = None

    try:
        name, target_layer = find_last_conv2d(model)
        print(f"CAM target: {name}")
        has_conv = True
    except ValueError:
        print("No Conv2d found")
        has_conv     = False
        target_layer = None

    _orig_gc = explainer.get_gradcam_explanations

    def new_gradcam(images, target_layers=None, method='gradcam'):
        if not has_conv: return _orig_gc(images, target_layers, method)
        if len(images.shape) == 3: images = images.unsqueeze(0)
        return [get_gradcam_explanation(model, images, target_layer)]

    def new_gradcam_pp(images):
        if not has_conv: return np.zeros((224, 224), dtype=np.float32)
        if len(images.shape) == 3: images = images.unsqueeze(0)
        return get_gradcam_plus_plus(model, images, target_layer)

    def new_integrated_gradients(images, steps=64):
        if len(images.shape) == 3: images = images.unsqueeze(0)
        return get_integrated_gradients(model, images, steps=steps)

    def new_rise(images, num_masks=6000, p=0.25, mask_size=5):
        if len(images.shape) == 3: images = images.unsqueeze(0)
        return get_rise_explanation(model, images, num_masks=num_masks, p=p, mask_size=mask_size)

    def new_gradient_shap(images, background_samples=None, n_samples=50, **kw):
        bg = explainer._shap_bg
        if bg is None:
            raise RuntimeError("GradientSHAP background missing. Provide train_loader to patch_explainer().")
        if len(images.shape) == 3: images = images.unsqueeze(0)
        return get_gradient_shap_explanation(model, images, bg, n_samples=n_samples)

    def new_lime(image, num_samples=300, num_features=10):
        return get_lime_explanation(model, image, device, num_samples, num_features, mean_t, std_t)

    explainer.get_gradcam_explanations   = new_gradcam
    explainer.get_gradcam_plus_plus      = new_gradcam_pp
    explainer.get_integrated_gradients   = new_integrated_gradients
    explainer.get_rise_explanations      = new_rise
    explainer.get_shap_explanations      = new_gradient_shap
    explainer.get_lime_explanations      = new_lime

    print("Patched: GradCAM, GradCAM++, IntegratedGradients, RISE , GradientSHAP, LIME")


if __name__ == '__main__':
    pass