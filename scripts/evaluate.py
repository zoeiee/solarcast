import os
import argparse
import json
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import Config, get_device
from ml.datasets.solar_dataset import SolarDataset
from ml.models.solar_cnn import SolarCNN
from ml.evaluation.metrics import compute_binary_metrics, format_metrics_report
from ml.evaluation.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

def main():
    parser = argparse.ArgumentParser(description="Evaluate SolarCast models")
    parser.add_argument('--model-path', type=str, default='models/best_model.pth')
    parser.add_argument('--manifest', type=str, default='data/processed/test_manifest.csv')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    
    if not os.path.exists(args.manifest):
        print(f"Test manifest not found: {args.manifest}")
        return
        
    print(f"Evaluating model {args.model_path} on {args.manifest}")
    dataset = SolarDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = SolarCNN().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Warning: Model weights not found at {args.model_path}, using randomly initialized model for demo.")
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device).float()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    y_true = np.array(all_labels).flatten()
    y_prob = np.array(all_preds).flatten()
    
    metrics = compute_binary_metrics(y_true, y_prob)
    
    # Save results
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(format_metrics_report(metrics))
    
    # Generate plots
    from sklearn.metrics import confusion_matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plot_confusion_matrix(cm, ['No Flare', 'Flare'], os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_prob, os.path.join(args.output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_prob, os.path.join(args.output_dir, 'pr_curve.png'))
    
    print(f"Saved evaluation results and plots to {args.output_dir}")

if __name__ == '__main__':
    main()
