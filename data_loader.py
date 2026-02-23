import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from config import Config
import glob
from tqdm import tqdm

class COVID19RadiographyDataset(Dataset):
    """
    Dataset class for COVID-19 Radiography Database
    Handles automatic train/val/test splitting since the dataset comes as a whole
    """
    def __init__(self, root_dir, split='train', transform=None, split_ratio=None):
    
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(Config.CLASSES)}
        
        if split_ratio is None:
            split_ratio = {
                'train': Config.TRAIN_RATIO,
                'val': Config.VAL_RATIO,
                'test': Config.TEST_RATIO
            }
        
        self._load_and_split_data(split_ratio)
    
    def _load_and_split_data(self, split_ratio):
        """
        Load all data and create train/val/test splits
        """
        print(f"Loading COVID-19 Radiography Database for {self.split} split...")
        
        all_images = []
        all_labels = []
        
        # Load all images from each class folder
        for class_name in Config.CLASSES:
            # Handle different possible folder names
            possible_names = [class_name, class_name.replace('_', ' ')]
            class_dir = None
            
            for name in possible_names:
                potential_dir = os.path.join(self.root_dir, name)
                if os.path.exists(potential_dir):
                    class_dir = potential_dir
                    break
            
            if class_dir and os.path.exists(class_dir):
                # Look for images in the class directory
                # The dataset structure might be:
                # - COVID/images/*.png or just COVID/*.png
                
                # Check for images subdirectory
                images_dir = os.path.join(class_dir, 'images')
                if os.path.exists(images_dir):
                    search_dir = images_dir
                else:
                    search_dir = class_dir
                
                # Find all image files
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    image_files.extend(glob.glob(os.path.join(search_dir, ext)))
                
                # Add to lists
                all_images.extend(image_files)
                all_labels.extend([self.class_to_idx[class_name]] * len(image_files))
                
                print(f"  Found {len(image_files)} images for class '{class_name}'")
        
        if len(all_images) == 0:
            print(f"WARNING: No images found in {self.root_dir}")
            print(f"Expected folder structure: {self.root_dir}/[COVID|Normal|Viral Pneumonia]/[images/]*.png")
            return
        
        # Convert to numpy arrays for splitting
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        # Perform stratified split
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_images, all_labels,
            test_size=split_ratio['test'],
            random_state=Config.SEED,
            stratify=all_labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = split_ratio['val'] / (split_ratio['train'] + split_ratio['val'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=Config.SEED,
            stratify=y_temp
        )
        
        # Assign the appropriate split
        if self.split == 'train':
            self.images = X_train.tolist()
            self.labels = y_train.tolist()
        elif self.split == 'val':
            self.images = X_val.tolist()
            self.labels = y_val.tolist()
        elif self.split == 'test':
            self.images = X_test.tolist()
            self.labels = y_test.tolist()
        
        print(f"  {self.split} set: {len(self.images)} images")
        
        # Print class distribution for this split
        unique, counts = np.unique(self.labels, return_counts=True)
        for idx, count in zip(unique, counts):
            class_name = Config.CLASSES[idx]
            print(f"    {class_name}: {count} images ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Convert to numpy array for albumentations
        image_np = np.array(image)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed['image']
        
        return image, label

class DataManager:
    def __init__(self):
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        
    def get_transforms(self):
        """Get data transformations"""
        from preprocessing import get_train_transforms, get_val_transforms
        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()
        self.test_transform = get_val_transforms()
    
    def get_data_loaders(self):
        """
        Create data loaders for COVID-19 Radiography Database
        """
        self.get_transforms()
        
        # Create datasets with automatic splitting
        train_dataset = COVID19RadiographyDataset(
            Config.DATA_DIR, 
            split='train', 
            transform=self.train_transform
        )
        
        val_dataset = COVID19RadiographyDataset(
            Config.DATA_DIR, 
            split='val', 
            transform=self.val_transform
        )
        
        test_dataset = COVID19RadiographyDataset(
            Config.DATA_DIR, 
            split='test', 
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch for training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_dataset(self, data_dir=None):
        """
        Analyze the COVID-19 Radiography Database
        """
        if data_dir is None:
            data_dir = Config.DATA_DIR
        
        class_counts = {}
        total_images = 0
        
        print("\n" + "="*50)
        print("COVID-19 Radiography Database Analysis")
        print("="*50)
        print(f"Dataset location: {data_dir}")
        
        for class_name in Config.CLASSES:
            # Handle different possible folder names
            possible_names = [class_name, class_name.replace('_', ' ')]
            class_dir = None
            
            for name in possible_names:
                potential_dir = os.path.join(data_dir, name)
                if os.path.exists(potential_dir):
                    class_dir = potential_dir
                    break
            
            if class_dir and os.path.exists(class_dir):
                # Check for images subdirectory
                images_dir = os.path.join(class_dir, 'images')
                if os.path.exists(images_dir):
                    search_dir = images_dir
                else:
                    search_dir = class_dir
                
                # Count images
                count = 0
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    count += len(glob.glob(os.path.join(search_dir, ext)))
                
                class_counts[class_name] = count
                total_images += count
        
        if total_images > 0:
            print(f"\nTotal images: {total_images}")
            print("\nClass Distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / total_images * 100)
                print(f"  {class_name}: {count} images ({percentage:.2f}%)")
            
            # Calculate imbalance ratio
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
            
            # Suggest class weights
            print("\nSuggested Class Weights (inverse frequency):")
            for class_name, count in class_counts.items():
                weight = total_images / (len(class_counts) * count)
                print(f"  {class_name}: {weight:.2f}")
        else:
            print("WARNING: No images found!")
            print(f"Please ensure the dataset is extracted to: {data_dir}")
            print("Expected structure:")
            print("  COVID-19_Radiography_Dataset/")
            print("    ├── COVID/")
            print("    │   └── images/ (or *.png files directly)")
            print("    ├── Normal/")
            print("    │   └── images/ (or *.png files directly)")
            print("    └── Viral Pneumonia/")
            print("        └── images/ (or *.png files directly)")
        
        return class_counts
    
    def get_class_weights(self):
        """
        Calculate balanced class weights for the dataset
        """
        # Create a temporary dataset to get all labels
        temp_dataset = COVID19RadiographyDataset(
            Config.DATA_DIR, 
            split='train', 
            transform=None
        )
        
        labels = np.array(temp_dataset.labels)
        
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        print("\nComputed Class Weights:")
        for idx, weight in enumerate(class_weights):
            print(f"  {Config.CLASSES[idx]}: {weight:.3f}")
        
        return torch.FloatTensor(class_weights)
    
    def verify_data_integrity(self):
        """
        Verify that images can be loaded and check for corrupted files
        """
        print("\n" + "="*50)
        print("Verifying Data Integrity")
        print("="*50)
        
        corrupted_files = []
        total_files = 0
        
        for class_name in Config.CLASSES:
            possible_names = [class_name, class_name.replace('_', ' ')]
            class_dir = None
            
            for name in possible_names:
                potential_dir = os.path.join(Config.DATA_DIR, name)
                if os.path.exists(potential_dir):
                    class_dir = potential_dir
                    break
            
            if class_dir and os.path.exists(class_dir):
                images_dir = os.path.join(class_dir, 'images')
                if os.path.exists(images_dir):
                    search_dir = images_dir
                else:
                    search_dir = class_dir
                
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    image_files.extend(glob.glob(os.path.join(search_dir, ext)))
                
                print(f"\nChecking {len(image_files)} images in {class_name}...")
                for img_path in tqdm(image_files, desc=f"  {class_name}"):
                    total_files += 1
                    try:
                        img = Image.open(img_path)
                        img.verify()  # Verify image integrity
                    except Exception as e:
                        corrupted_files.append((img_path, str(e)))
        
        if corrupted_files:
            print(f"\nFound {len(corrupted_files)} corrupted files:")
            for file_path, error in corrupted_files[:5]:  # Show first 5
                print(f"  {file_path}: {error}")
        else:
            print(f"\n✓ All {total_files} images verified successfully!")
        
        return corrupted_files

if __name__ == "__main__":
    # Test the data loader
    dm = DataManager()
    
    # Analyze dataset
    dm.analyze_dataset()
    
    # Verify data integrity
    dm.verify_data_integrity()
    
    # Get class weights
    weights = dm.get_class_weights()
    
    # Test loading
    print("\nTesting data loaders...")
    train_loader, val_loader, test_loader = dm.get_data_loaders()
    
    print(f"\nData Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nSample batch:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {target.shape}")
        print(f"  Labels: {target[:10]}")
        break