import warnings
warnings.filterwarnings('ignore')
from config import Config
from data_loader import DataManager
from models import get_model
from training import Trainer
from evaluation import ModelEvaluator
import torch

def safe_quick_run():
    Config.set_seed(Config.SEED)
    print("="*60)
    print("MEMORY-SAFE QUICK TEST")
    print("="*60)
    
    dm = DataManager()
    train_loader, val_loader, test_loader = dm.get_data_loaders()
    
    print("\n[1/3] Training model (5 epochs)...")
    model = get_model('ensemble')
    model.to(Config.DEVICE)
    
    trainer = Trainer(model, train_loader, val_loader)
    trained_model, history = trainer.train(epochs=5)
    
    print("\n[2/3] Evaluating model...")
    evaluator = ModelEvaluator(trained_model, test_loader)
    results = evaluator.evaluate()
    evaluator.print_results()
    
    print("\n[3/3] Testing XAI (limited samples)...")
    from advanced_xai import EnhancedXAIExplainer
    explainer = EnhancedXAIExplainer(trained_model)
    
    mem_eval = fixes['MemoryEfficientEvaluator'](
        explainer, 
        batch_size=2, 
        max_scales=2
    )
    
    test_images = []
    for batch_idx, (data, _) in enumerate(test_loader):
        test_images.extend([data[i] for i in range(min(2, data.size(0)))])
        if len(test_images) >= 4:
            break
    
    xai_results = mem_eval.evaluate_batch(test_images[:4], method='simple')
    print(f"Generated {len(xai_results)} XAI explanations")
    
    print("\n" + "="*60)
    print("SAFE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return {
        'model_results': results,
        'xai_samples': len(xai_results)
    }

if __name__ == "__main__":
    results = safe_quick_run()
