import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from .loss import compute_loss

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                epochs: int = 200, 
                lr: float = 1e-3, 
                weight_decay: float = 1e-5,
                device: str = 'cpu') -> dict:
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for z_batch, x_true_batch in train_loader:
            z_batch = z_batch.to(device)
            x_true_batch = x_true_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: KalmanFormer produces sequence of estimates
            x_est_seq = model(z_seq=z_batch)
            
            # Loss
            loss = compute_loss(x_est_seq, x_true_batch, model, lambda_reg=0.0) # Reg handled by optimizer
            
            loss.backward()
            
            # Gradient Clipping optional but good for RNNs/Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for z_val, x_val in val_loader:
                z_val = z_val.to(device)
                x_val = x_val.to(device)
                x_est_val = model(z_seq=z_val)
                loss_val = compute_loss(x_est_val, x_val, model, lambda_reg=0.0)
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
    return history
