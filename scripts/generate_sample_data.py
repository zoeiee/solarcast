import os
import numpy as np
import pandas as pd
from PIL import Image

def generate_sample_data(output_dir: str, num_samples: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_data = []
    
    for i in range(num_samples):
        # Even indices: Quiet (label 0), Odd indices: Flare-like (label 1)
        label = i % 2
        
        if label == 0:
            # Quiet: Gaussian noise + weak dipole
            img = np.random.normal(0, 50, (128, 128))
            y, x = np.ogrid[-64:64, -64:64]
            mask1 = x**2 + y**2 <= 15**2
            mask2 = (x-20)**2 + y**2 <= 15**2
            img[mask1] += 150
            img[mask2] -= 150
            img = np.clip(img, -200, 200)
        else:
            # Flare: Strong bipolar region with steep gradient
            img = np.random.normal(0, 100, (128, 128))
            y, x = np.ogrid[-64:64, -64:64]
            mask1 = (x+5)**2 + y**2 <= 20**2
            mask2 = (x-5)**2 + y**2 <= 20**2
            img[mask1] += 800
            img[mask2] -= 800
            img = np.clip(img, -1000, 1000)
            
        filename = f"magnetogram_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, img)
        
        # Also save as PNG for demo purposes
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_norm)
        img_pil.save(os.path.join(output_dir, f"demo_{i:03d}.png"))
        
        manifest_data.append({
            'filepath': filepath,
            'label': label,
            'active_region_id': f"AR{10000 + i // 4}",
            'timestamp': f"2023-01-01T{i:02d}:00:00"
        })
        
    df = pd.DataFrame(manifest_data)
    df.to_csv(os.path.join(output_dir, 'manifest.csv'), index=False)

if __name__ == '__main__':
    generate_sample_data('data/sample')
