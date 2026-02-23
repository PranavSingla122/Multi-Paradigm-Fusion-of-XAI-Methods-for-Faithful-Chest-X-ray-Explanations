import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

def hybrid_contrast_enhancement(img, **params):

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Lambda(image=hybrid_contrast_enhancement, name='hybrid_contrast'),
        A.HorizontalFlip(p=Config.AUGMENTATION_PARAMS['horizontal_flip_prob']),
        A.RandomBrightnessContrast(p=Config.AUGMENTATION_PARAMS['brightness_contrast_prob']),
        A.ShiftScaleRotate(
            shift_limit=Config.AUGMENTATION_PARAMS['shift_limit'],
            scale_limit=Config.AUGMENTATION_PARAMS['scale_limit'],
            rotate_limit=Config.AUGMENTATION_PARAMS['rotation_limit'],
            p=0.3
        ),
        A.GaussNoise(
            var_limit=Config.AUGMENTATION_PARAMS['noise_var_limit'],
            p=Config.AUGMENTATION_PARAMS['noise_prob']
        ),
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Lambda(image=hybrid_contrast_enhancement, name='hybrid_contrast'),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_multi_scale_transforms(scale):
    return A.Compose([
        A.Resize(scale, scale),
        A.Lambda(image=hybrid_contrast_enhancement, name='hybrid_contrast'),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class DataAugmentor:
    def __init__(self):
        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()
    
    def apply_smote(self, X_flat, y):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=Config.SEED)
        X_resampled, y_resampled = smote.fit_resample(X_flat, y)
        return X_resampled, y_resampled
    
    def apply_mixup(self, data, targets, alpha=1.0):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        
        lam = np.random.beta(alpha, alpha)
        
        mixed_data = lam * data + (1 - lam) * shuffled_data
        targets_a, targets_b = targets, shuffled_targets
        
        return mixed_data, targets_a, targets_b, lam
    
    def apply_cutmix(self, data, targets, beta=1.0):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        
        lam = np.random.beta(beta, beta)
        
        batch_size = data.size(0)
        h, w = data.size(2), data.size(3)
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        data[:, :, bby1:bby2, bbx1:bbx2] = shuffled_data[:, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return data, targets, shuffled_targets, lam

import torch

def visualize_augmentations(original_image, num_augmentations=6):
    import matplotlib.pyplot as plt
    
    augmentor = DataAugmentor()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_augmentations):
        if i == 0:
            axes[i].imshow(original_image)
            axes[i].set_title('Original')
        else:
            augmented = augmentor.train_transform(image=np.array(original_image))['image']
            augmented_np = augmented.permute(1, 2, 0).numpy()
            augmented_np = (augmented_np - augmented_np.min()) / (augmented_np.max() - augmented_np.min())
            axes[i].imshow(augmented_np)
            axes[i].set_title(f'Augmentation {i}')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.VISUALIZATION_DIR}/augmentation_samples.png', dpi=150)
    plt.show()