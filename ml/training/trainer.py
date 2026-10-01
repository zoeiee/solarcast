import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Tuple
import os
import random
from tqdm import tqdm
from ..evaluation.metrics import compute_binary_metrics
from ..config import Config, get_device
from .losses import get_loss_fn

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SolarTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Config):
        set_seed(config.get('seed', 42))
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.criterion = get_loss_fn(config).to(self.device)
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        scheduler_type = config.get('scheduler', 'reduce_on_plateau')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 50))
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
            
        self.history = {'train_loss': [], 'val_loss': [], 'val_tss': []}
        
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
        metrics = compute_binary_metrics(np.array(all_labels).flatten(), np.array(all_preds).flatten())
        return total_loss / len(self.train_loader), metrics

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device).float().unsqueeze(1)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
        metrics = compute_binary_metrics(np.array(all_labels).flatten(), np.array(all_preds).flatten())
        return total_loss / len(self.val_loader), metrics

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        best_tss = -1.0
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        save_dir = self.config.get('save_dir', 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val TSS: {val_metrics['tss']:.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_tss'].append(val_metrics['tss'])
            
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['tss'])
            else:
                self.scheduler.step()
                
            if val_metrics['tss'] > best_tss:
                best_tss = val_metrics['tss']
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                patience_counter = 0
                print("Saved best model.")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        return self.history
