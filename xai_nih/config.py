import torch
import os
import random
import numpy as np

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    DATA_DIR = r'/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
    
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation_studies')
    STATISTICAL_DIR = os.path.join(RESULTS_DIR, 'statistical_analysis')
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(ABLATION_DIR, exist_ok=True)
    os.makedirs(STATISTICAL_DIR, exist_ok=True)
    CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    NUM_CLASSES = 14  
    
    MULTI_LABEL = True
    CLASS_WEIGHTS = None
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 64  
    NUM_WORKERS = 8
    
    
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8
    
    PREDICTION_THRESHOLD = 0.4
    POS_WEIGHT_MULTIPLIER = 1.2
    USE_CLASS_BALANCED_LOSS = False
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = 0.25
    
    LABEL_SMOOTHING = 0.0
    MIXUP_ALPHA = 0.0
    CUTMIX_ALPHA = 0.0
    MIXUP_PROB = 0.0
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    AUGMENTATION_PARAMS = {
        'horizontal_flip_prob': 0.5,
        'brightness_contrast_prob': 0.4,
        'rotation_limit': 20,
        'shift_limit': 0.15,
        'scale_limit': 0.15,
        'noise_var_limit': (10.0, 50.0),
        'noise_prob': 0.3,
        'gaussian_blur_prob': 0.2,
        'clahe_prob': 0.5
    }
    
    CLAHE_PARAMS = {
        'clip_limit': 2.0,
        'tile_grid_size': (8, 8)
    }
    
    DROPOUT_RATE = 0.2
    MC_SAMPLES = 100
    
    SHAP_BACKGROUND_SAMPLES = 50
    LIME_NUM_SAMPLES = 300
    
    ENHANCED_XAI_PARAMS = {
        'multi_scale_sizes': [224, 336, 448, 560, 672],
        'fusion_methods': ['adaptive', 'pyramid', 'attention', 'confidence_weighted', 'simple'],
        'gaussian_blur_kernel': 5,
        'gaussian_blur_sigma': 1.5,
        'adaptive_threshold_percentile': 0.7,
        'noise_levels': [0.05, 0.1, 0.15, 0.2],
        'integrated_gradients_steps': 50,
        'rise_num_masks': 8000,
        'rise_mask_probability': 0.5
    }
    
    XAI_EVALUATION_PARAMS = {
        'faithfulness_steps': 20,
        'faithfulness_methods': ['insertion', 'deletion'],
        'stability_perturbations': 20,
        'stability_noise_levelss': [0.05, 0.1, 0.15, 0.2],
        'consistency_num_samples': 50,
        'min_faithfulness_target': 0.70,
        'min_stability_target': 0.45
    }
    
    CROSS_DATASET_PARAMS = {
        'datasets': ['COVID_Radiography', 'COVIDx_CXR4'],
        'bidirectional_validation': True,
        'domain_shift_metrics': ['mmd', 'coral', 'wasserstein'],
        'max_performance_drop': 0.15,
        'min_samples_per_dataset': 1000
    }
    
    STATISTICAL_PARAMS = {
        'confidence_level': 0.95,
        'alpha': 0.01,
        'bonferroni_correction': True,
        'bootstrap_iterations': 10000,
        'effect_size_methods': ['cohens_d', 'glass_delta', 'hedges_g'],
        'paired_tests': ['wilcoxon', 't_test'],
        'independent_tests': ['mann_whitney', 'welch_t']
    }
    
    ABLATION_PARAMS = {
        'scale_combinations': [
            [224],
            [224, 448],
            [224, 336, 448],
            [224, 336, 448, 560],
            [224, 336, 448, 560, 672]
        ],
        'component_ablation': {
            'attention': True,
            'dropout': True,
            'preprocessing': True,
            'ensemble_fusion': True,
            'multi_scale': True
        },
        'hyperparameter_ranges': {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'gaussian_sigma': [0.5, 1.0, 1.5, 2.0, 2.5]
        },
        'num_trials_per_config': 1,
        'ablation_epochs': 20,
        'ablation_patience': 5
    }
    

    
    EXPERIMENT_TRACKING = {
        'log_interval': 10,
        'save_checkpoint_every': 5,
        'track_gpu_memory': True,
        'track_inference_time': True,
        'save_predictions': True,
        'save_explanations': True
    }
    
    COMPUTATIONAL_EFFICIENCY = {
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 2,
        'pin_memory': True,
        'num_workers': 4,
        'prefetch_factor': 2
    }
    
    K_FOLDS = 5
    CONFIDENCE_INTERVAL_ALPHA = 0.95
    
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-7
    
    SEED = 42

    @staticmethod
    def set_seed(seed: int = 42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

Config.set_seed(Config.SEED)

