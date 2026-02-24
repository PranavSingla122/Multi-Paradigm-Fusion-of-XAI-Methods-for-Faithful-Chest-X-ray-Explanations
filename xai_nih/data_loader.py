import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from config import Config
from tqdm import tqdm

class NIHChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, split_ratio=None, max_samples=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        if split_ratio is None:
            split_ratio = {
                'train': Config.TRAIN_RATIO,
                'val': Config.VAL_RATIO,
                'test': Config.TEST_RATIO
            }
        
        self._load_and_split_data(split_ratio, max_samples)
    
    def _load_and_split_data(self, split_ratio, max_samples):
        print(f"Loading NIH Chest X-ray Database for {self.split} split...")
        
        csv_locations = [
            os.path.join(self.root_dir, 'Data_Entry_2017.csv'),
            os.path.join(self.root_dir, '3', 'Data_Entry_2017.csv'),
            os.path.join(self.root_dir, '1', 'Data_Entry_2017.csv'),
            os.path.join(self.root_dir, '2', 'Data_Entry_2017.csv'),
        ]
        
        csv_path = None
        for loc in csv_locations:
            if os.path.exists(loc):
                csv_path = loc
                print(f"Found CSV at: {csv_path}")
                break
        
        if csv_path is None:
            print(f"ERROR: Data_Entry_2017.csv not found!")
            return
        
        df = pd.read_csv(csv_path)
        print(f"Total entries in CSV: {len(df)}")
        
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=Config.SEED)
            print(f"Limited to {len(df)} samples")
        
        csv_base = os.path.dirname(csv_path)
        
        all_images = []
        all_labels = []
        
        print("Building image path index...")
        image_paths = {}
        for root, dirs, files in os.walk(csv_base):
            for file in files:
                if file.endswith('.png'):
                    image_paths[file] = os.path.join(root, file)
        
        print(f"Indexed {len(image_paths)} images")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
            img_name = row['Image Index']
            finding_labels = row['Finding Labels']
            
            if img_name not in image_paths:
                continue
            
            img_path = image_paths[img_name]
            
            label_vector = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            
            if finding_labels == 'No Finding':
                label_vector[-1] = 1.0
            else:
                diseases = finding_labels.split('|')
                for disease in diseases:
                    if disease in Config.CLASSES:
                        label_idx = Config.CLASSES.index(disease)
                        label_vector[label_idx] = 1.0
            
            all_images.append(img_path)
            all_labels.append(label_vector)
        
        print(f"Found {len(all_images)} valid images")
        
        if len(all_images) == 0:
            print("WARNING: No images found!")
            return
        
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_images, all_labels,
            test_size=split_ratio['test'],
            random_state=Config.SEED,
            stratify=all_labels[:, -1]
        )
        
        val_size_adjusted = split_ratio['val'] / (split_ratio['train'] + split_ratio['val'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=Config.SEED,
            stratify=y_temp[:, -1]
        )
        
        if self.split == 'train':
            self.images = X_train.tolist()
            self.labels = y_train.tolist()
        elif self.split == 'val':
            self.images = X_val.tolist()
            self.labels = y_val.tolist()
        elif self.split == 'test':
            self.images = X_test.tolist()
            self.labels = y_test.tolist()
        
        print(f"{self.split} set: {len(self.images)} images")
        
        if len(self.labels) > 0:
            labels_array = np.array(self.labels)
            for idx, class_name in enumerate(Config.CLASSES):
                count = labels_array[:, idx].sum()
                print(f"  {class_name}: {int(count)} images ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        max_retries = 10
        for retry in range(max_retries):
            try:
                img_path = self.images[idx]
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                
                if self.transform:
                    transformed = self.transform(image=image_np)
                    image = transformed['image']
                
                return image, label
                
            except (OSError, IOError):
                idx = (idx + 1) % len(self.images)
                if retry == max_retries - 1:
                    return torch.zeros(3, 224, 224), label
    
    def get_sample_weights(self):
        labels = np.array(self.labels)
        pos_counts = labels.sum(axis=0)
        
        weights = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            label_indices = np.where(labels[i] == 1)[0]
            if len(label_indices) > 0:
                label_weights = 1.0 / (pos_counts[label_indices] + 1e-5)
                weights[i] = label_weights.mean()
            else:
                weights[i] = 1.0
        
        weights = weights / weights.sum() * len(weights)
        return weights

class DataManager:
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def get_torch_generator(seed=None):
        if seed is None:
            seed = Config.SEED
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    def __init__(self):
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        Config.set_seed(Config.SEED)
        
    def get_transforms(self):
        from preprocessing import get_train_transforms, get_val_transforms
        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()
        self.test_transform = get_val_transforms()
    
    def get_data_loaders(self, max_samples=None, use_weighted_sampling=True):
        self.get_transforms()
        
        train_dataset = NIHChestXrayDataset(
            Config.DATA_DIR, 
            split='train', 
            transform=self.train_transform,
            max_samples=max_samples
        )
        
        val_dataset = NIHChestXrayDataset(
            Config.DATA_DIR, 
            split='val', 
            transform=self.val_transform,
            max_samples=max_samples
        )
        
        test_dataset = NIHChestXrayDataset(
            Config.DATA_DIR, 
            split='test', 
            transform=self.test_transform,
            max_samples=max_samples
        )
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty! Check your data directory and CSV file.")
        
        if use_weighted_sampling:
            sample_weights = train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=self.get_torch_generator()
            )
            shuffle = False
            print("✓ Using WeightedRandomSampler for balanced training")
        else:
            sampler = None
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.get_torch_generator(),
            drop_last=True if len(train_dataset) > Config.BATCH_SIZE else False,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.get_torch_generator(),
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.get_torch_generator(),
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )
        
        labels = np.array(train_dataset.labels)
        pos_counts = labels.sum(axis=0)
        neg_counts = len(labels) - pos_counts
        
        pos_weights = np.sqrt(neg_counts / (pos_counts + 1e-5))
        pos_weights = np.clip(pos_weights, 1.0, 8.0) * Config.POS_WEIGHT_MULTIPLIER
        pos_weights = torch.FloatTensor(pos_weights).to(Config.DEVICE)
        
        print(f"\nClass Distribution:")
        for i, class_name in enumerate(Config.CLASSES):
            print(f"  {class_name:20s}: {int(pos_counts[i]):6d} ({pos_counts[i]/len(labels)*100:5.2f}%) | weight: {pos_weights[i]:.2f}")
        
        return train_loader, val_loader, test_loader, pos_weights
    
    def analyze_dataset(self):
        print("\n" + "="*50)
        print("NIH Chest X-ray Database Analysis")
        print("="*50)
        
        csv_locations = [
            os.path.join(Config.DATA_DIR, 'Data_Entry_2017.csv'),
            os.path.join(Config.DATA_DIR, '3', 'Data_Entry_2017.csv'),
            os.path.join(Config.DATA_DIR, '1', 'Data_Entry_2017.csv'),
            os.path.join(Config.DATA_DIR, '2', 'Data_Entry_2017.csv'),
        ]
        
        csv_path = None
        for loc in csv_locations:
            if os.path.exists(loc):
                csv_path = loc
                break
        
        if csv_path is None:
            print(f"ERROR: Data_Entry_2017.csv not found!")
            return {}
        
        print(f"Using CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Total images: {len(df)}")
        
        print("\nDisease Distribution:")
        disease_counts = {}
        for disease in Config.CLASSES:
            if disease == 'No Finding':
                count = (df['Finding Labels'] == 'No Finding').sum()
            else:
                count = df['Finding Labels'].str.contains(disease, na=False).sum()
            disease_counts[disease] = count
            print(f"  {disease}: {count} ({count/len(df)*100:.2f}%)")
        
        return disease_counts
    
    def get_pos_weights(self):
        temp_dataset = NIHChestXrayDataset(
            Config.DATA_DIR, 
            split='train', 
            transform=None,
            max_samples=10000
        )
        
        if len(temp_dataset.labels) == 0:
            print("WARNING: No labels found, using uniform weights")
            return torch.ones(Config.NUM_CLASSES)
        
        labels = np.array(temp_dataset.labels)
        pos_counts = labels.sum(axis=0)
        neg_counts = len(labels) - pos_counts
        
        pos_weights = np.sqrt(neg_counts / (pos_counts + 1e-5))
        pos_weights = np.clip(pos_weights, 1.0, 8.0) * Config.POS_WEIGHT_MULTIPLIER
        
        print("\nPositive Class Weights (clipped to max 8):")
        for idx, weight in enumerate(pos_weights):
            print(f"  {Config.CLASSES[idx]}: {weight:.3f}")
        
        return torch.FloatTensor(pos_weights)
    
    def get_class_weights(self):
        return self.get_pos_weights()

if __name__ == "__main__":
    dm = DataManager()
    dm.analyze_dataset()
    weights = dm.get_pos_weights()
    
    print("\nTesting data loaders...")
    train_loader, val_loader, test_loader, pos_weights = dm.get_data_loaders(max_samples=1000)
    
    print(f"\nData Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nSample batch:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {target.shape}")
        print(f"  Sample labels: {target[0]}")
        break
