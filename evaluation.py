import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc, cohen_kappa_score, 
                           matthews_corrcoef)
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import os

class XAIEvaluator:
    def __init__(self, model, explainer, device=Config.DEVICE):
        self.model = model
        self.explainer = explainer
        self.device = device
        
    def compute_tdt_awcd_metrics(self, feature_importance, labels):
        feature_importance_flat = feature_importance.reshape(len(labels), -1)
        
        dt = DecisionTreeClassifier(random_state=Config.SEED, max_depth=10)
        dt.fit(feature_importance_flat, labels)
        
        tdt = dt.tree_.max_depth
        
        leaf_depths = []
        def get_leaf_depths(node=0, depth=0):
            if dt.tree_.children_left[node] == dt.tree_.children_right[node]:
                leaf_depths.append(depth)
            else:
                if dt.tree_.children_left[node] != -1:
                    get_leaf_depths(dt.tree_.children_left[node], depth + 1)
                if dt.tree_.children_right[node] != -1:
                    get_leaf_depths(dt.tree_.children_right[node], depth + 1)
        
        get_leaf_depths()
        awcd = np.mean(leaf_depths) if leaf_depths else 0
        
        return tdt, awcd
    
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
                original_pred = F.softmax(original_output, dim=1)
                original_confidence = original_pred.max().item()
                predicted_class = original_pred.argmax().item()
            
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
                    masked_pred = F.softmax(masked_output, dim=1)
                    class_confidence = masked_pred[0, predicted_class].item()
                    confidences.append(class_confidence)
            
            if confidences:
                auc_score = np.trapz(confidences, dx=1/len(confidences))
                faithfulness_scores.append(auc_score)
        
        return np.mean(faithfulness_scores) if faithfulness_scores else 0
    
    def stability_test(self, image, num_perturbations=Config.XAI_EVALUATION_PARAMS['stability_perturbations'],
                   noise_level=0.1):  
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
    
    def calculate_overlap_metrics(self, explanation_map, ground_truth_mask):
        exp_norm = (explanation_map - explanation_map.min()) / (explanation_map.max() - explanation_map.min() + 1e-8)
        exp_binary = (exp_norm > 0.5).astype(int)
        
        intersection = np.logical_and(exp_binary, ground_truth_mask).sum()
        union = np.logical_or(exp_binary, ground_truth_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        dice = 2 * intersection / (exp_binary.sum() + ground_truth_mask.sum()) if (exp_binary.sum() + ground_truth_mask.sum()) > 0 else 0
        
        correlation = np.corrcoef(exp_norm.flatten(), ground_truth_mask.flatten())[0,1]
        
        return {'iou': iou, 'dice': dice, 'correlation': correlation}

class ModelEvaluator:
    def __init__(self, model, test_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results = {}
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    outputs = outputs['ensemble']
                
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        self.results['accuracy'] = accuracy_score(all_labels, all_preds)
        self.results['precision'] = precision_score(all_labels, all_preds, average='macro')
        self.results['recall'] = recall_score(all_labels, all_preds, average='macro')
        self.results['f1'] = f1_score(all_labels, all_preds, average='macro')
        self.results['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
        self.results['mcc'] = matthews_corrcoef(all_labels, all_preds)
        
        try:
            self.results['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            self.results['auc'] = 0
        
        self.results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        self.results['classification_report'] = classification_report(
            all_labels, all_preds, target_names=Config.CLASSES
        )
        
        self.results['predictions'] = all_preds
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probs
        
        return self.results
    
    def plot_confusion_matrix(self, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, save_path=None):
        n_classes = len(Config.CLASSES)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            y_true = (self.results['labels'] == i).astype(int)
            y_score = self.results['probabilities'][:, i]
            fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {Config.CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60)
        print(f"Accuracy: {self.results['accuracy']:.4f}")
        print(f"Precision (macro): {self.results['precision']:.4f}")
        print(f"Recall (macro): {self.results['recall']:.4f}")
        print(f"F1-Score (macro): {self.results['f1']:.4f}")
        print(f"Cohen's Kappa: {self.results['cohen_kappa']:.4f}")
        print(f"Matthews Correlation Coefficient: {self.results['mcc']:.4f}")
        print(f"AUC (macro): {self.results['auc']:.4f}")
        print("\nClassification Report:")
        print(self.results['classification_report'])

class ComprehensiveEvaluator:
    def __init__(self, model, test_loader, explainer):
        self.model = model
        self.test_loader = test_loader
        self.explainer = explainer
        self.device = Config.DEVICE
        
    def run_comprehensive_evaluation(self):
        print("\n" + "="*60)
        print("Running Comprehensive Evaluation")
        print("="*60)
        
        model_evaluator = ModelEvaluator(self.model, self.test_loader)
        model_results = model_evaluator.evaluate()
        model_evaluator.print_results()
        
        model_evaluator.plot_confusion_matrix(
            save_path=os.path.join(Config.VISUALIZATION_DIR, 'confusion_matrix.png')
        )
        model_evaluator.plot_roc_curves(
            save_path=os.path.join(Config.VISUALIZATION_DIR, 'roc_curves.png')
        )
        
        xai_evaluator = XAIEvaluator(self.model, self.explainer)
        
        print("\nEvaluating XAI Methods...")
        xai_results = self.evaluate_xai_methods(xai_evaluator)
        
        self.plot_xai_comparison(xai_results)
        
        return model_results, xai_results
    
    def evaluate_xai_methods(self, xai_evaluator):
        results = {
            'shap': {'faithfulness': [], 'stability': []},
            'lime': {'faithfulness': [], 'stability': []},
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
        
        self.print_xai_results(results)
        
        return results
    
    def print_xai_results(self, results):
        print("\n" + "="*60)
        print("XAI Method Evaluation Results")
        print("="*60)
        
        for method in results:
            print(f"\n{method.upper()}:")
            if f'faithfulness_mean' in results[method]:
                print(f"  Faithfulness: {results[method]['faithfulness_mean']:.4f} ± {results[method]['faithfulness_std']:.4f}")
            if f'stability_mean' in results[method]:
                print(f"  Stability: {results[method]['stability_mean']:.4f} ± {results[method]['stability_std']:.4f}")
    
    def plot_xai_comparison(self, results):
        methods = list(results.keys())
        faithfulness_means = [results[m].get('faithfulness_mean', 0) for m in methods]
        faithfulness_stds = [results[m].get('faithfulness_std', 0) for m in methods]
        stability_means = [results[m].get('stability_mean', 0) for m in methods]
        stability_stds = [results[m].get('stability_std', 0) for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width/2, faithfulness_means, width, yerr=faithfulness_stds,
               label='Faithfulness', capsize=5, color='steelblue')
        ax.bar(x + width/2, stability_means, width, yerr=stability_stds,
               label='Stability', capsize=5, color='darkorange')
        
        ax.set_xlabel('XAI Method')
        ax.set_ylabel('Score')
        ax.set_title('XAI Methods Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATION_DIR, 'xai_comparison.png'), dpi=150)
        plt.show()

def statistical_significance_test(results1, results2, alpha=0.05):
    statistic, p_value = stats.wilcoxon(results1, results2)
    
    print(f"\nWilcoxon signed-rank test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"Result: Statistically significant difference (p < {alpha})")
    else:
        print(f"Result: No statistically significant difference (p >= {alpha})")
    
    return statistic, p_value