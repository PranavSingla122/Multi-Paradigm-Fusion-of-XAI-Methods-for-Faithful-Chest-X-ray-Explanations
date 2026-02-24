import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
from config import Config
from models import get_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.clip is not None and self.clip > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y) 
            pt1 = pt1 + self.clip
            los_pos = los_pos * (1 - pt0)**self.gamma_pos
            los_neg = los_neg * pt1**self.gamma_neg

        loss = -los_pos - los_neg
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()


class EnsembleLoss(nn.Module):
    """Custom loss for improved ensemble with deep supervision"""
    def __init__(self, num_classes, base_criterion, aux_weight=0.3, consistency_weight=0.1):
        super(EnsembleLoss, self).__init__()
        self.num_classes = num_classes
        self.aux_weight = aux_weight
        self.consistency_weight = consistency_weight
        
        self.main_criterion = base_criterion
        self.aux_criterion = base_criterion
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, outputs, targets):
        # Main ensemble loss
        main_loss = self.main_criterion(outputs['logits'], targets)
        
        total_loss = main_loss
        loss_dict = {'main_loss': main_loss.item()}
        
        # Auxiliary losses (deep supervision) - only if present
        if 'vit_aux' in outputs:
            vit_aux_loss = self.aux_criterion(outputs['vit_aux'], targets)
            swin_aux_loss = self.aux_criterion(outputs['swin_aux'], targets)
            convnext_aux_loss = self.aux_criterion(outputs['convnext_aux'], targets)
            
            aux_loss = (vit_aux_loss + swin_aux_loss + convnext_aux_loss) / 3
            total_loss += self.aux_weight * aux_loss
            
            loss_dict['aux_loss'] = aux_loss.item()
            
            # Consistency loss (KL divergence between auxiliary and main predictions)
            if self.consistency_weight > 0:
                # For multi-label, use sigmoid; for single-label, use softmax
                if Config.MULTI_LABEL:
                    main_probs = torch.sigmoid(outputs['logits'])
                    vit_probs = torch.sigmoid(outputs['vit_aux'])
                    swin_probs = torch.sigmoid(outputs['swin_aux'])
                    convnext_probs = torch.sigmoid(outputs['convnext_aux'])
                else:
                    main_probs = F.log_softmax(outputs['logits'], dim=1)
                    vit_probs = F.softmax(outputs['vit_aux'], dim=1)
                    swin_probs = F.softmax(outputs['swin_aux'], dim=1)
                    convnext_probs = F.softmax(outputs['convnext_aux'], dim=1)
                
                consistency_loss = (
                    self.kl_div(main_probs, vit_probs) +
                    self.kl_div(main_probs, swin_probs) +
                    self.kl_div(main_probs, convnext_probs)
                ) / 3
                
                total_loss += self.consistency_weight * consistency_loss
                loss_dict['consistency_loss'] = consistency_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Trainer:
    def __init__(self, model, train_loader, val_loader, pos_weights=None, device=Config.DEVICE, use_ablation_settings=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Check if model is improved ensemble
        self.is_improved_ensemble = hasattr(model, 'use_aux_loss') and model.use_aux_loss
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=Config.WARMUP_EPOCHS
        )
        
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS - Config.WARMUP_EPOCHS,
            eta_min=Config.MIN_LR
        )
        
        # Setup base criterion
        if Config.MULTI_LABEL:
            if Config.USE_CLASS_BALANCED_LOSS:
                base_criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
                print(f"✓ Using AsymmetricLoss (gamma_neg=2, gamma_pos=1)")
            else:
                base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                print(f"✓ Using BCEWithLogitsLoss with pos_weights (multiplier={Config.POS_WEIGHT_MULTIPLIER})")
        else:
            from data_loader import DataManager
            dm = DataManager()
            class_weights = dm.get_class_weights().to(device)
            base_criterion = FocalLoss(alpha=1, gamma=2, pos_weight=class_weights)
        
        # Wrap with ensemble loss if using improved ensemble
        if self.is_improved_ensemble:
            self.criterion = EnsembleLoss(
                num_classes=Config.NUM_CLASSES,
                base_criterion=base_criterion,
                aux_weight=0.3,
                consistency_weight=0.1
            )
            print(f"✓ Using EnsembleLoss with Deep Supervision (aux_weight=0.3, consistency=0.1)")
        else:
            self.criterion = base_criterion
        
        if torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        patience = Config.ABLATION_PARAMS.get('ablation_patience', Config.EARLY_STOPPING_PATIENCE) if use_ablation_settings else Config.EARLY_STOPPING_PATIENCE
        
        self.early_stopping = EarlyStopping(
            patience=patience,
            verbose=True
        )
        
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'val_auroc': [],
            'per_class_metrics': [],
            'learning_rate': [],
            'loss_components': []  # Track auxiliary losses
        }
        
        self.best_auroc = 0
        self.best_f1 = 0
        self.best_acc = 0
        self.current_epoch = 0
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        loss_components_epoch = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    logits = outputs.get('ensemble', outputs.get('logits', outputs))
                else:
                    logits = outputs
                
                # Calculate loss
                if self.is_improved_ensemble and isinstance(outputs, dict):
                    loss, loss_dict = self.criterion(outputs, target)
                    loss_components_epoch.append(loss_dict)
                else:
                    loss = self.criterion(logits, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    continue
                
                if Config.MULTI_LABEL:
                    predictions = (torch.sigmoid(logits) > Config.PREDICTION_THRESHOLD).float()
                else:
                    predictions = logits.argmax(dim=1)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if (batch_idx + 1) % Config.COMPUTATIONAL_EFFICIENCY['gradient_accumulation_steps'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if (batch_idx + 1) % Config.COMPUTATIONAL_EFFICIENCY['gradient_accumulation_steps'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(self.train_loader)
        
        # Save loss components
        if loss_components_epoch:
            avg_components = {}
            for key in loss_components_epoch[0].keys():
                avg_components[key] = np.mean([d[key] for d in loss_components_epoch])
            self.training_history['loss_components'].append(avg_components)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if Config.MULTI_LABEL:
            accuracy = (all_preds == all_labels).mean()
            precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    logits = outputs.get('ensemble', outputs.get('logits', outputs))
                else:
                    logits = outputs
                
                # Calculate loss
                if self.is_improved_ensemble and isinstance(outputs, dict):
                    loss, _ = self.criterion(outputs, target)
                else:
                    loss = self.criterion(logits, target)
                
                if Config.MULTI_LABEL:
                    probs = torch.sigmoid(logits)
                    predictions = (probs > Config.PREDICTION_THRESHOLD).float()
                else:
                    probs = torch.softmax(logits, dim=1)
                    predictions = logits.argmax(dim=1)
                
                running_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        avg_loss = running_loss / len(self.val_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        if Config.MULTI_LABEL:
            accuracy = (all_preds == all_labels).mean()
            precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
            
            try:
                auroc = roc_auc_score(all_labels, all_probs, average='macro')
            except:
                auroc = 0.5
            
            per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_auroc = []
            per_class_sens = []
            per_class_spec = []
            
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
                
                per_class_sens.append(sens)
                per_class_spec.append(spec)
            
            per_class_metrics = {
                'f1': per_class_f1,
                'precision': per_class_precision,
                'recall': per_class_recall,
                'auroc': per_class_auroc,
                'sensitivity': per_class_sens,
                'specificity': per_class_spec
            }
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            auroc = 0.5
            per_class_metrics = {}
        
        return avg_loss, accuracy, precision, recall, f1, auroc, per_class_metrics
    
    def train(self, epochs=Config.EPOCHS):
        print(f"\nTraining on {Config.DEVICE}")
        print(f"Model Type: {'Improved Ensemble' if self.is_improved_ensemble else 'Standard'}")
        print(f"Threshold: {Config.PREDICTION_THRESHOLD}, Batch Size: {Config.BATCH_SIZE}")
        print(f"LR: {Config.LEARNING_RATE}, Weight Decay: {Config.WEIGHT_DECAY}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auroc, per_class = self.validate()
            
            if epoch < Config.WARMUP_EPOCHS:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.cosine_scheduler.step()
                current_lr = self.cosine_scheduler.get_last_lr()[0]
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['train_precision'].append(train_prec)
            self.training_history['train_recall'].append(train_rec)
            self.training_history['train_f1'].append(train_f1)
            
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_precision'].append(val_prec)
            self.training_history['val_recall'].append(val_rec)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['val_auroc'].append(val_auroc)
            self.training_history['per_class_metrics'].append(per_class)
            self.training_history['learning_rate'].append(current_lr)
            
            print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            print(f'Val   - Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, AUROC: {val_auroc:.4f}')
            
            # Print loss components for improved ensemble
            if self.training_history['loss_components'] and len(self.training_history['loss_components']) > epoch:
                components = self.training_history['loss_components'][epoch]
                if 'aux_loss' in components:
                    print(f'Loss  - Main: {components["main_loss"]:.4f}, Aux: {components["aux_loss"]:.4f}', end='')
                    if 'consistency_loss' in components:
                        print(f', Cons: {components["consistency_loss"]:.4f}')
                    else:
                        print()
            
            if Config.MULTI_LABEL and per_class:
                print("\nKey Diseases:")
                key_diseases = ['Pneumonia', 'Pneumothorax', 'Infiltration', 'Cardiomegaly']
                for disease in key_diseases:
                    if disease in Config.CLASSES:
                        idx = Config.CLASSES.index(disease)
                        print(f"  {disease:15s}: F1={per_class['f1'][idx]:.3f}, "
                              f"Prec={per_class['precision'][idx]:.3f}, "
                              f"Rec={per_class['recall'][idx]:.3f}, "
                              f"AUROC={per_class['auroc'][idx]:.3f}")
            
            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'auroc': val_auroc
                }, os.path.join(Config.MODEL_DIR, 'best_model_auroc.pth'))
                print(f'✓ New best AUROC model saved: {val_auroc:.4f}')
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'f1': val_f1
                }, os.path.join(Config.MODEL_DIR, 'best_model_f1.pth'))
                print(f'✓ New best F1 model saved: {val_f1:.4f}')
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'accuracy': val_acc
                }, os.path.join(Config.MODEL_DIR, 'best_model.pth'))
                print(f'✓ New best Accuracy model saved: {val_acc:.4f}')
            
            self.early_stopping(val_loss, self.model, 
                              os.path.join(Config.MODEL_DIR, 'checkpoint.pth'))
            
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            print(f'Learning rate: {current_lr:.6f}')
        
        self.plot_training_history()
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best AUROC: {self.best_auroc:.4f}")
        print(f"Best F1:    {self.best_f1:.4f}")
        print(f"Best Acc:   {self.best_acc:.4f}")
        
        return self.model, self.training_history
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(self.training_history['val_precision'], label='Precision')
        axes[0, 2].plot(self.training_history['val_recall'], label='Recall')
        axes[0, 2].plot(self.training_history['val_f1'], label='F1-Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Validation Metrics')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        if self.training_history['val_auroc']:
            axes[1, 0].plot(self.training_history['val_auroc'], label='Val AUROC', color='purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUROC')
            axes[1, 0].set_title('Validation AUROC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.training_history['train_f1'], label='Train F1')
        axes[1, 1].plot(self.training_history['val_f1'], label='Val F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        if self.training_history['learning_rate']:
            axes[1, 2].plot(self.training_history['learning_rate'], label='LR', linewidth=2, color='green')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].set_yscale('log')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATION_DIR, 'training_history.png'), dpi=150)
        plt.close()