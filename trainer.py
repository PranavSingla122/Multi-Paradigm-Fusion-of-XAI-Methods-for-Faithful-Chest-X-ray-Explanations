import torch
from config import Config
from data_loader import DataManager
from models import get_model
from training import Trainer

def main():
    print("Loading dataset...")
    data_manager = DataManager()
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print("\nInitializing model...")
    model = get_model('ensemble')
    model.to(Config.DEVICE)
    
    print("\nStarting training...")
    trainer = Trainer(model, train_loader, val_loader)
    trained_model, history = trainer.train(epochs=50)
    
    print("\nTraining complete!")
    print(f"Model saved to: {Config.MODEL_DIR}/best_model.pth")

if __name__ == "__main__":
    main()