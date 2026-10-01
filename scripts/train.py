import os
import argparse
import sys
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import Config
from ml.datasets.solar_dataset import SolarDataset
from ml.models.solar_cnn import SolarCNN
from ml.models.baseline import train_baseline
from ml.models.transfer_model import TransferSolarCNN
from ml.training.trainer import SolarTrainer

def main():
    parser = argparse.ArgumentParser(description="Train SolarCast models")
    parser.add_argument('--model', type=str, choices=['cnn', 'baseline', 'transfer'], default='cnn')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    
    if args.model == 'baseline':
        print("Training baseline model (RandomForest)...")
        features_csv = config.get('features_csv', 'data/processed/features.csv')
        if not os.path.exists(features_csv):
            print(f"Features file {features_csv} not found. Run preprocess_data.py first.")
            return
        df = pd.read_csv(features_csv)
        drop_cols = ['filepath', 'label', 'active_region_id']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df['label'].values
        
        model, metrics = train_baseline(X, y)
        print(f"Baseline training complete. Metrics: {metrics}")
        
    else:
        print(f"Training deep learning model: {args.model}")
        train_manifest = config.get('train_manifest', 'data/processed/train_manifest.csv')
        val_manifest = config.get('val_manifest', 'data/processed/val_manifest.csv')
        
        if not os.path.exists(train_manifest) or not os.path.exists(val_manifest):
            print("Manifest files not found. Run preprocess_data.py first.")
            return
            
        train_ds = SolarDataset(train_manifest)
        val_ds = SolarDataset(val_manifest)
        
        train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 16), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 16), shuffle=False)
        
        if args.model == 'cnn':
            model = SolarCNN()
        elif args.model == 'transfer':
            model = TransferSolarCNN()
            
        trainer = SolarTrainer(model, train_loader, val_loader, config)
        history = trainer.train(num_epochs=config.get('epochs', 10))
        print("Training complete.")

if __name__ == '__main__':
    main()
