import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing.image_processing import prepare_for_model
from ml.features.magnetic_features import extract_all_features
from ml.datasets.splits import active_region_split

def main():
    parser = argparse.ArgumentParser(description="Preprocess SolarCast data")
    parser.add_argument('--manifest', type=str, default='data/sample/manifest.csv', help='Input manifest CSV')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        return

    df = pd.read_csv(args.manifest)
    os.makedirs(args.output_dir, exist_ok=True)
    features_list = []
    
    print("Processing images and extracting features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filepath = row['filepath']
        if os.path.exists(filepath):
            img = np.load(filepath)
            # Extract features before full preprocessing (which alters original field values)
            features = extract_all_features(img)
            features['filepath'] = filepath
            features['label'] = row['label']
            features['active_region_id'] = row['active_region_id']
            features_list.append(features)
            
            # Optional: save preprocessed tensor directly
            # tensor = prepare_for_model(img)
            # torch.save(tensor, os.path.join(args.output_dir, f"tensor_{idx}.pt"))
            
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(os.path.join(args.output_dir, 'features.csv'), index=False)
    
    # Splits
    train_df, val_df, test_df = active_region_split(df)
    train_df.to_csv(os.path.join(args.output_dir, 'train_manifest.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val_manifest.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_manifest.csv'), index=False)
    
    print(f"Saved preprocessed data and splits to {args.output_dir}")

if __name__ == '__main__':
    main()
