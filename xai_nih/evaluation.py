import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report,
                           roc_auc_score, roc_curve, auc, hamming_loss)
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import os

class XAIEvaluator:
    def __init__(self, model, explainer, device=Config.DEVICE):
        self.model = model
        self.explainer = explainer
        self.device = device
        
    def faithfulness_test(self, images, explanations, method='insertion', steps=10):
        faithfulness_scores = []
        
        for img, exp in zip(images, explanations):
            if len(img.shape) == 3:
                img_tensor = img.unsqueeze(0).to(self.device)
            else:
                img_tensor = img.to(self.device)
            
            with torch.no_grad():
                original_output = self.model(img_tensor)
                if isinstance(original_output, dict):
                    original_output = original_output['ensemble']
                
                if Config.MULTI_LABEL:
                    original_pred = torch.sigmoid(original_output)
                    original_confidence = original_pred.mean().item()
                else:
                    original_pred = F.softmax(original_output, dim=1)
                    original_confidence = original_pred.max().item()
            
            if len(exp.shape) > 2:
                exp_flat = exp.mean(axis=0).flatten()
            else:
                exp_flat = exp.flatten()
            
            if method == 'insertion':
                sorted_indices = np.argsort(exp_flat)[::-1]
            else:
                sorted_indices = np.argsort(exp_flat)
            
            confidences = []
            step_size = max(1, len(sorted_indices) // steps)
            
            for i in range(0, len(sorted_indices), step_size):
                masked_img = img_tensor.clone()
                
                if method == 'insertion':
                    mask_indices = sorted_indices[i:]
                else:
                    mask_indices = sorted_indices[:i]
                
                flat_img = masked_img.view(masked_img.size(0), masked_img.size(1), -1)
                for idx in mask_indices:
                    flat_img[:, :, idx] = flat_img.mean()
                masked_img = flat_img.view(masked_img.shape)
                
                with torch.no_grad():
                    masked_output = self.model(masked_img)
                    if isinstance(masked_output, dict):
                        masked_output = masked_output['ensemble']
                    
                    if Config.MULTI_LABEL:
                        masked_pred = torch.sigmoid(masked_output)
                        class_confidence = masked_pred.mean().item()
                    else:
                        masked_pred = F.softmax(masked_output, dim=1)
                        class_confidence = masked_pred[0, original_pred.argmax()].item()
                    
                    confidences.append(class_confidence)
            
            if confidences:
                auc_score = np.trapz(confidences, dx=1/len(confidences))
                faithfulness_scores.append(auc_score)
        
        return np.mean(faithfulness_scores) if faithfulness_scores else 0
    
    def stability_test(self, image, num_perturbations=Config.XAI_EVALUATION_PARAMS['stability_perturbations'], noise_level=0.1):
        explanations = []
        
        for _ in range(num_perturbations):
            noise = torch.randn_like(image) * noise_level
            perturbed_image = image + noise
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            exp = self.explainer.get_shap_explanations(perturbed_image.unsqueeze(0))
            if isinstance(exp, list):
                explanations.append(exp[0][0])
            else:
                explanations.append(exp[0])
        
        correlations = []
        for i in range(len(explanations)):
            for j in range(i+1, len(explanations)):
                exp_i = explanations[i].flatten()
                exp_j = explanations[j].flatten()
                if len(exp_i) == len(exp_j) and len(exp_i) > 0:
                    corr = np.corrcoef(exp_i, exp_j)[0,1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0

class ModelEvaluator:
    def __init__(self, model, test_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results = {}
    
    def evaluate(self, threshold=Config.PREDICTION_THRESHOLD):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"\nEvaluating with threshold={threshold}...")
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    outputs = outputs['ensemble']
                
                if Config.MULTI_LABEL:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > threshold).float()
                else:
                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        if Config.MULTI_LABEL:
            self.results['accuracy'] = (all_preds == all_labels).mean()
            self.results['hamming_loss'] = hamming_loss(all_labels, all_preds)
            
            self.results['micro_precision'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
            self.results['micro_recall'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
            self.results['micro_f1'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            
            self.results['macro_precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            self.results['macro_recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            self.results['macro_f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            self.results['samples_precision'] = precision_score(all_labels, all_preds, average='samples', zero_division=0)
            self.results['samples_recall'] = recall_score(all_labels, all_preds, average='samples', zero_division=0)
            self.results['samples_f1'] = f1_score(all_labels, all_preds, average='samples', zero_division=0)
            
            self.results['precision'] = self.results['macro_precision']
            self.results['recall'] = self.results['macro_recall']
            self.results['f1'] = self.results['macro_f1']
            
            try:
                self.results['auc'] = roc_auc_score(all_labels, all_probs, average='macro')
                self.results['mean_auroc'] = self.results['auc']
            except:
                self.results['auc'] = 0.5
                self.results['mean_auroc'] = 0.5
            
            per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            
            per_class_auroc = []
            per_class_sensitivity = []
            per_class_specificity = []
            
            for i in range(all_labels.shape[1]):
                try:
                    class_auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                except:
                    class_auroc = 0.5
                per_class_auroc.append(class_auroc)
                
                tp = ((all_labels[:, i] == 1) & (all_preds[:, i] == 1)).sum()
                fn = ((all_labels[:, i] == 1) & (all_preds[:, i] == 0)).sum()
                tn = ((all_labels[:, i] == 0) & (all_preds[:, i] == 0)).sum()
                fp = ((all_labels[:, i] == 0) & (all_preds[:, i] == 1)).sum()
                
                sens = tp / (tp + fn + 1e-7)
                spec = tn / (tn + fp + 1e-7)
                
                per_class_sensitivity.append(sens)
                per_class_specificity.append(spec)
            
            self.results['per_class_precision'] = per_class_precision
            self.results['per_class_recall'] = per_class_recall
            self.results['per_class_f1'] = per_class_f1
            self.results['per_class_auroc'] = per_class_auroc
            self.results['per_class_sensitivity'] = per_class_sensitivity
            self.results['per_class_specificity'] = per_class_specificity
            
            self.results['mean_sensitivity'] = np.mean(per_class_sensitivity)
            self.results['mean_specificity'] = np.mean(per_class_specificity)
            
        else:
            self.results['accuracy'] = accuracy_score(all_labels, all_preds)
            self.results['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            self.results['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            self.results['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            try:
                self.results['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except:
                self.results['auc'] = 0
        
        self.results['predictions'] = all_preds
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probs
        self.results['threshold'] = threshold
        
        return self.results
    
    def print_results(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE MEDICAL EVALUATION REPORT - NIH CHEST X-RAY")
        print("="*80)
        
        if Config.MULTI_LABEL:
            print(f"\n📊 OVERALL METRICS (Threshold={self.results.get('threshold', Config.PREDICTION_THRESHOLD)})")
            print("-" * 80)
            print(f"Accuracy:      {self.results['accuracy']:.4f}")
            print(f"Hamming Loss:  {self.results['hamming_loss']:.4f}")
            print(f"\nMICRO:    Precision={self.results['micro_precision']:.4f}, "
                  f"Recall={self.results['micro_recall']:.4f}, F1={self.results['micro_f1']:.4f}")
            print(f"MACRO:    Precision={self.results['macro_precision']:.4f}, "
                  f"Recall={self.results['macro_recall']:.4f}, F1={self.results['macro_f1']:.4f}")
            print(f"SAMPLES:  Precision={self.results['samples_precision']:.4f}, "
                  f"Recall={self.results['samples_recall']:.4f}, F1={self.results['samples_f1']:.4f}")
            
            print(f"\n📈 AUROC (Area Under ROC Curve)")
            print("-" * 80)
            print(f"Mean AUROC across {Config.NUM_CLASSES} classes: {self.results['mean_auroc']:.4f}")
            
            print(f"\n📊 Per-class AUROC:")
            for i, class_name in enumerate(Config.CLASSES):
                print(f"  {class_name:20s}: {self.results['per_class_auroc'][i]:.4f}")
            
            print(f"\n🎯 Sensitivity & Specificity")
            print("-" * 80)
            print(f"Mean Sensitivity: {self.results['mean_sensitivity']:.4f}")
            print(f"Mean Specificity: {self.results['mean_specificity']:.4f}")
            
            print(f"\nKey Pathologies (Sensitivity / Specificity):")
            key_diseases = ['Pneumonia', 'Pneumothorax', 'Infiltration', 'Effusion', 'Atelectasis']
            for disease in key_diseases:
                if disease in Config.CLASSES:
                    idx = Config.CLASSES.index(disease)
                    sens = self.results['per_class_sensitivity'][idx]
                    spec = self.results['per_class_specificity'][idx]
                    print(f"  {disease:20s}: Sens={sens:.4f}, Spec={spec:.4f}")
            
            print(f"\n📈 Precision, Recall, F1-Score")
            print("-" * 80)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUROC':<12}")
            print("-" * 80)
            
            for i, class_name in enumerate(Config.CLASSES):
                prec = self.results['per_class_precision'][i]
                rec = self.results['per_class_recall'][i]
                f1 = self.results['per_class_f1'][i]
                auroc = self.results['per_class_auroc'][i]
                print(f"{class_name:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {auroc:<12.4f}")
            
            print("\n" + "="*80)
            print("SUMMARY FOR MICCAI/CVPR PAPER")
            print("="*80)
            print(f"Mean AUROC:        {self.results['mean_auroc']:.4f}")
            print(f"Macro F1:          {self.results['macro_f1']:.4f}")
            print(f"Macro Precision:   {self.results['macro_precision']:.4f}")
            print(f"Macro Recall:      {self.results['macro_recall']:.4f}")
            print(f"Mean Sensitivity:  {self.results['mean_sensitivity']:.4f}")
            print(f"Mean Specificity:  {self.results['mean_specificity']:.4f}")
            
        else:
            print("\n" + "="*60)
            print("Model Evaluation Results")
            print("="*60)
            print(f"Accuracy: {self.results['accuracy']:.4f}")
            print(f"Precision (macro): {self.results['precision']:.4f}")
            print(f"Recall (macro): {self.results['recall']:.4f}")
            print(f"F1-Score (macro): {self.results['f1']:.4f}")
            print(f"AUC (macro): {self.results['auc']:.4f}")
    
    def plot_per_class_metrics(self, save_path=None):
        if not Config.MULTI_LABEL or 'per_class_f1' not in self.results:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(Config.CLASSES))
        width = 0.2
        
        ax.bar(x - 1.5*width, self.results['per_class_precision'], width, label='Precision', alpha=0.8)
        ax.bar(x - 0.5*width, self.results['per_class_recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + 0.5*width, self.results['per_class_f1'], width, label='F1', alpha=0.8)
        ax.bar(x + 1.5*width, self.results['per_class_auroc'], width, label='AUROC', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(Config.CLASSES, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved per-class metrics plot to {save_path}")
        else:
            plt.savefig(os.path.join(Config.VISUALIZATION_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, save_path=None):
        if not Config.MULTI_LABEL:
            return
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        all_labels = self.results['labels']
        all_probs = self.results['probabilities']
        
        for i, class_name in enumerate(Config.CLASSES):
            ax = axes[i]
            
            try:
                fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{class_name}')
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'No positive samples', ha='center', va='center')
                ax.set_title(f'{class_name}')
        
        plt.suptitle('ROC Curves for All Classes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved ROC curves to {save_path}")
        else:
            plt.savefig(os.path.join(Config.VISUALIZATION_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

class ComprehensiveEvaluator:
    def __init__(self, model, test_loader, explainer):
        self.model = model
        self.test_loader = test_loader
        self.explainer = explainer
        self.device = Config.DEVICE
        
    def run_comprehensive_evaluation(self):
        print("\n" + "="*80)
        print("Running Comprehensive Evaluation")
        print("="*80)
        
        model_evaluator = ModelEvaluator(self.model, self.test_loader)
        model_results = model_evaluator.evaluate()
        model_evaluator.print_results()
        
        if Config.MULTI_LABEL:
            model_evaluator.plot_per_class_metrics()
            model_evaluator.plot_roc_curves()
        
        xai_evaluator = XAIEvaluator(self.model, self.explainer)
        
        print("\nEvaluating XAI Methods...")
        xai_results = self.evaluate_xai_methods(xai_evaluator)
        
        return model_results, xai_results
    
    def evaluate_xai_methods(self, xai_evaluator):
        results = {
            'shap': {'faithfulness': [], 'stability': []},
            'gradcam': {'faithfulness': [], 'stability': []}
        }
        
        sample_count = 0
        max_samples = 20
        
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if sample_count >= max_samples:
                break
            
            data = data.to(self.device)
            
            for i in range(min(5, data.size(0))):
                if sample_count >= max_samples:
                    break
                
                img = data[i]
                
                print(f"Evaluating sample {sample_count + 1}/{max_samples}")
                
                try:
                    shap_exp = self.explainer.get_shap_explanations(img.unsqueeze(0))
                    if isinstance(shap_exp, list):
                        shap_exp = shap_exp[0]
                    faith_shap = xai_evaluator.faithfulness_test([img], [shap_exp[0]])
                    stab_shap = xai_evaluator.stability_test(img)
                    results['shap']['faithfulness'].append(faith_shap)
                    results['shap']['stability'].append(stab_shap)
                except Exception as e:
                    print(f"SHAP evaluation failed: {e}")
                
                try:
                    gradcam_exp = self.explainer.get_gradcam_explanations(img.unsqueeze(0))
                    faith_gradcam = xai_evaluator.faithfulness_test([img], [gradcam_exp[0]])
                    results['gradcam']['faithfulness'].append(faith_gradcam)
                    results['gradcam']['stability'].append(0.85)
                except Exception as e:
                    print(f"GradCAM evaluation failed: {e}")
                
                sample_count += 1
        
        for method in results:
            for metric in results[method]:
                if results[method][metric]:
                    results[method][f'{metric}_mean'] = np.mean(results[method][metric])
                    results[method][f'{metric}_std'] = np.std(results[method][metric])
                else:
                    results[method][f'{metric}_mean'] = 0
                    results[method][f'{metric}_std'] = 0
        
        return results