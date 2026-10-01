import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from typing import Optional, Callable, Dict, Any

class SolarDataset(Dataset):
    def __init__(self, manifest_csv: str, transform: Optional[Callable] = None):
        self.manifest = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        filepath = row['filepath']
        label = int(row['label'])
        
        # Metadata
        metadata = {
            'active_region_id': row.get('active_region_id', 'unknown'),
            'timestamp': row.get('timestamp', 'unknown')
        }
        
        if not os.path.exists(filepath):
            # Graceful handling of missing files - return zero tensor
            image_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
        else:
            try:
                img = np.load(filepath)
                # Expand dims to make it 1-channel
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                image_tensor = torch.from_numpy(img).float()
            except Exception:
                image_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
                
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return {
            'image': image_tensor,
            'label': label,
            'metadata': metadata
        }
