import torch
import os
import random
import numpy as np

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_DIR = r'/teamspace/studios/this_studio/.cache/kagglehub/datasets/tawsifurrahman/covid19-radiography-database/versions/5/COVID-19_Radiography_Dataset'''
        
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
    
    CLASSES = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
    NUM_CLASSES = len(CLASSES)
    CLASS_WEIGHTS = [2.8, 1.0, 1.7, 7.5]
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    AUGMENTATION_PARAMS = {
        'horizontal_flip_prob': 0.5,
        'brightness_contrast_prob': 0.3,
        'rotation_limit': 15,
        'shift_limit': 0.1,
        'scale_limit': 0.1,
        'noise_var_limit': (10.0, 50.0),
        'noise_prob': 0.2
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
        'ablation_epochs': 30,
        'ablation_patience': 5
    }
    AUGMENTATION_PARAMS = {
    'horizontal_flip_prob': 0.5,
    'brightness_contrast_prob': 0.3,
    'shift_limit': 0.05,
    'scale_limit': 0.05,
    'rotation_limit': 10,
    'noise_var_limit': (10, 50),
    'noise_prob': 0.2
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
        'gradient_accumulation_steps': 1,
        'pin_memory': True,
        'num_workers': 4,
        'prefetch_factor': 2
    }
    
    K_FOLDS = 5
    CONFIDENCE_INTERVAL_ALPHA = 0.95

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


