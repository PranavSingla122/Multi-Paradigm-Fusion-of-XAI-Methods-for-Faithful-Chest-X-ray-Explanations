import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Dict, List
import json
import os
from sklearn.metrics import roc_auc_score
from config import Config
from statistical_analysis import StatisticalAnalyzer
from models import get_model
from training import Trainer
from evaluation import ModelEvaluator


class ModelOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._logged_keys = False

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            if not self._logged_keys:
                print(f"  [Debug] Model output keys: {list(out.keys())}")
                self._logged_keys = True
            for k in ('ensemble', 'logits', 'predictions', 'output', 'out'):
                if k in out:
                    return out[k]
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    return v
        raise ValueError(f"Cannot extract logits from {type(out)}")


class BaselineArchitectures:
    @staticmethod
    def get_resnet():
        m = models.resnet152(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, Config.NUM_CLASSES)
        return m

    @staticmethod
    def get_efficientnet():
        m = models.efficientnet_b7(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, Config.NUM_CLASSES)
        return m

    @staticmethod
    def get_densenet():
        m = models.densenet201(pretrained=False)
        m.classifier = nn.Linear(m.classifier.in_features, Config.NUM_CLASSES)
        return m

    @staticmethod
    def get_vit_base():
        return get_model('vit')

    @staticmethod
    def get_swin_base():
        return get_model('swin')

    @staticmethod
    def get_convnext_base():
        return get_model('convnext')


class AblationStudy:
    def __init__(self, base_model_class, train_loader, val_loader, test_loader):
        self.base_model_class     = base_model_class
        self.train_loader         = train_loader
        self.val_loader           = val_loader
        self.test_loader          = test_loader
        self.results              = {}
        self.statistical_analyzer = StatisticalAnalyzer()

        self.baseline_models = {
            'ResNet-152':       BaselineArchitectures.get_resnet,
            'EfficientNet-B7':  BaselineArchitectures.get_efficientnet,
            'DenseNet-201':     BaselineArchitectures.get_densenet,
            'ViT-Base':         BaselineArchitectures.get_vit_base,
            'Swin-Transformer': BaselineArchitectures.get_swin_base,
            'ConvNeXt-Base':    BaselineArchitectures.get_convnext_base,
        }

    def count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def compute_auroc(self, model: nn.Module, loader) -> float:
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(Config.DEVICE)
                out    = model(inputs)
                probs  = torch.sigmoid(out)  # NIH uses sigmoid (multi-label)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_probs  = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        try:
            # Per-class AUROC then macro average for multi-label NIH
            aurocs = []
            for c in range(Config.NUM_CLASSES):
                if all_labels[:, c].sum() > 0:
                    aurocs.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
            return float(np.mean(aurocs)) if aurocs else 0.0
        except Exception as e:
            print(f"  Warning: AUROC failed: {e}")
            return 0.0

    def train_and_evaluate_model(self, model_fn, model_name: str,
                                  num_trials: int = 3, epochs: int = 50) -> Dict:
        print(f"\n{'='*70}\nTraining {model_name} ({epochs} epochs, {num_trials} trials)\n{'='*70}")

        accuracies, f1_scores, aurocs, losses = [], [], [], []
        param_count = None

        for trial in range(num_trials):
            print(f"\n  Trial {trial+1}/{num_trials}")
            model = model_fn() if callable(model_fn) else model_fn
            model.to(Config.DEVICE)

            if param_count is None:
                param_count = self.count_parameters(model)
                print(f"  Parameters: {param_count:,}")

            trainer = Trainer(model, self.train_loader, self.val_loader)
            trained_model, _ = trainer.train(epochs=epochs)

            wrapped   = ModelOutputWrapper(trained_model)
            evaluator = ModelEvaluator(wrapped, self.test_loader)
            res       = evaluator.evaluate()
            auroc     = self.compute_auroc(wrapped, self.test_loader)

            accuracies.append(res['accuracy'])
            f1_scores.append(res['f1'])
            aurocs.append(auroc)
            losses.append(res.get('loss', 0.0))

            print(f"    Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUROC: {auroc:.4f}")

        def stats(vals):
            return {
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'min':  float(np.min(vals)),  'max': float(np.max(vals)),
                'values': [float(v) for v in vals]
            }

        return {
            'model_name':      model_name,
            'parameter_count': param_count,
            'num_trials':      num_trials,
            'epochs':          epochs,
            'accuracy':        stats(accuracies),
            'f1_score':        stats(f1_scores),
            'auroc':           stats(aurocs),
            'loss':            stats(losses),
        }

    def run_complete_ablation_study(self, num_trials: int = None, epochs: int = None) -> Dict:
        if num_trials is None:
            num_trials = Config.ABLATION_PARAMS.get('num_trials_per_config', 3)
        if epochs is None:
            epochs = 50  # same as main training

        print("\n" + "="*80)
        print("NIH BASELINE ARCHITECTURE COMPARISON")
        print("="*80)
        print(f"Epochs: {epochs} | Trials: {num_trials} | Classes: {Config.NUM_CLASSES}")
        print(f"Baselines: ResNet-152, EfficientNet-B7, DenseNet-201, ViT-Base, Swin-Transformer, ConvNeXt-Base")
        print("="*80)

        all_results = {}

        all_results['Ensemble (Ours)'] = self.train_and_evaluate_model(
            self.base_model_class, "Ensemble (Ours)", num_trials=num_trials, epochs=epochs
        )

        for model_name, model_fn in self.baseline_models.items():
            all_results[model_name] = self.train_and_evaluate_model(
                model_fn, model_name, num_trials=num_trials, epochs=epochs
            )

        all_results['statistical_comparison'] = self._perform_statistical_comparison(all_results)

        self._save_results(all_results)
        self._generate_ablation_report(all_results)

        return all_results

    def _perform_statistical_comparison(self, results: Dict) -> Dict:
        acc, f1, auroc = {}, {}, {}
        for name, res in results.items():
            if name == 'statistical_comparison' or not isinstance(res, dict) or 'accuracy' not in res:
                continue
            acc[name]   = res['accuracy']['values']
            f1[name]    = res['f1_score']['values']
            auroc[name] = res['auroc']['values']

        return {
            'accuracy': self.statistical_analyzer.comprehensive_comparison(acc,   metric_name='accuracy'),
            'f1_score': self.statistical_analyzer.comprehensive_comparison(f1,    metric_name='f1_score'),
            'auroc':    self.statistical_analyzer.comprehensive_comparison(auroc,  metric_name='auroc'),
            'best_models': {
                'accuracy': max(acc,   key=lambda n: np.mean(acc[n])),
                'f1_score': max(f1,    key=lambda n: np.mean(f1[n])),
                'auroc':    max(auroc, key=lambda n: np.mean(auroc[n])),
            }
        }

    def _save_results(self, results: Dict):
        os.makedirs(Config.ABLATION_DIR, exist_ok=True)
        path = os.path.join(Config.ABLATION_DIR, 'nih_baseline_comparison_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"\nResults saved: {path}")

    def _generate_ablation_report(self, results: Dict):
        report  = "# NIH Baseline Architecture Comparison\n\n"
        report += f"**Epochs**: 50 (same for all models) | **Classes**: {Config.NUM_CLASSES} (multi-label)\n\n"
        report += "## Summary\n\n"
        report += "| Model | Parameters | Accuracy | F1-Score | AUROC |\n"
        report += "|-------|-----------|----------|----------|-------|\n"

        for name, res in results.items():
            if name == 'statistical_comparison' or not isinstance(res, dict) or 'accuracy' not in res:
                continue
            params = f"{res['parameter_count']:,}" if res.get('parameter_count') else 'N/A'
            report += (f"| {name} | {params} "
                       f"| {res['accuracy']['mean']:.4f}±{res['accuracy']['std']:.4f} "
                       f"| {res['f1_score']['mean']:.4f}±{res['f1_score']['std']:.4f} "
                       f"| {res['auroc']['mean']:.4f}±{res['auroc']['std']:.4f} |\n")

        if 'statistical_comparison' in results:
            bm = results['statistical_comparison'].get('best_models', {})
            report += "\n## Best Models\n\n"
            report += f"- **Accuracy**: {bm.get('accuracy','N/A')}\n"
            report += f"- **F1-Score**: {bm.get('f1_score','N/A')}\n"
            report += f"- **AUROC**: {bm.get('auroc','N/A')}\n\n"

        report += "\n## Parameter Efficiency\n\n"
        report += "| Model | Params | Accuracy | Params/Accuracy |\n"
        report += "|-------|--------|----------|-----------------|\n"
        for name, res in results.items():
            if name == 'statistical_comparison' or not isinstance(res, dict) or 'accuracy' not in res:
                continue
            p   = res.get('parameter_count', 0) or 0
            acc = res['accuracy']['mean']
            eff = p / acc if acc > 0 else float('inf')
            report += f"| {name} | {p:,} | {acc:.4f} | {eff:,.0f} |\n"

        path = os.path.join(Config.ABLATION_DIR, 'nih_ablation_report.md')
        with open(path, 'w') as f:
            f.write(report)
        print(f"Report saved: {path}")
        print("\n" + "="*80)
        print("NIH ABLATION STUDY COMPLETE")
        print("="*80)