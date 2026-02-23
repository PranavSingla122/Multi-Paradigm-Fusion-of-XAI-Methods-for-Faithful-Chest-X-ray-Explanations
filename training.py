import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
from config import Config
from models import FocalLoss, get_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

class Trainer:
    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS
        )
        
        from data_loader import DataManager
        dm = DataManager()
        class_weights = dm.get_class_weights().to(device)
        self.criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
        
        if torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            verbose=True
        )
        
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    loss = self.criterion(outputs['ensemble'], target)
                    predictions = outputs['ensemble'].argmax(dim=1)
                else:
                    loss = self.criterion(outputs, target)
                    predictions = outputs.argmax(dim=1)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    loss = self.criterion(outputs['ensemble'], target)
                    predictions = outputs['ensemble'].argmax(dim=1)
                else:
                    loss = self.criterion(outputs, target)
                    predictions = outputs.argmax(dim=1)
                
                running_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, epochs=Config.EPOCHS):
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()
            
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
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(Config.MODEL_DIR, 'best_model.pth'))
                print(f'New best model saved with accuracy: {val_acc:.4f}')
            
            self.early_stopping(val_loss, self.model, 
                              os.path.join(Config.MODEL_DIR, 'checkpoint.pth'))
            
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            self.scheduler.step()
            print(f'Learning rate: {self.scheduler.get_last_lr()[0]:.6f}')
        
        self.plot_training_history()
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
        
        axes[1, 0].plot(self.training_history['train_precision'], label='Train')
        axes[1, 0].plot(self.training_history['val_precision'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.training_history['train_recall'], label='Train')
        axes[1, 1].plot(self.training_history['val_recall'], label='Val')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(self.training_history['train_f1'], label='Train')
        axes[1, 2].plot(self.training_history['val_f1'], label='Val')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].set_title('F1-Score Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATION_DIR, 'training_history.png'), dpi=150)
        plt.show()

class KFoldTrainer:
    def __init__(self, model_class, k_folds=Config.K_FOLDS):
        self.model_class = model_class
        self.k_folds = k_folds
        self.fold_results = []
    
    def train_fold(self, fold_idx, train_loader, val_loader):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{self.k_folds}")
        print('='*50)
        
        model = self.model_class()
        trainer = Trainer(model, train_loader, val_loader)
        trained_model, history = trainer.train(epochs=Config.EPOCHS)
        
        return trained_model, history
    
    def cross_validate(self, dataset):
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Subset
        
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=Config.SEED)
        
        all_indices = list(range(len(dataset)))
        all_labels = [dataset[i][1] for i in all_indices]
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_indices, all_labels)):
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, 
                                     shuffle=True, num_workers=Config.NUM_WORKERS)
            val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, 
                                   shuffle=False, num_workers=Config.NUM_WORKERS)
            
            model, history = self.train_fold(fold_idx, train_loader, val_loader)
            
            torch.save(model.state_dict(), 
                      os.path.join(Config.MODEL_DIR, f'fold_{fold_idx}_best_model.pth'))
            
            self.fold_results.append({
                'fold': fold_idx,
                'history': history,
                'best_val_acc': max(history['val_acc']),
                'best_val_f1': max(history['val_f1'])
            })
        
        self.print_summary()
        return self.fold_results
    
    def print_summary(self):
        print("\n" + "="*50)
        print("K-Fold Cross Validation Summary")
        print("="*50)
        
        accuracies = [result['best_val_acc'] for result in self.fold_results]
        f1_scores = [result['best_val_f1'] for result in self.fold_results]
        
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        print("\nPer-Fold Results:")
        for result in self.fold_results:
            print(f"Fold {result['fold'] + 1}: Acc={result['best_val_acc']:.4f}, F1={result['best_val_f1']:.4f}")