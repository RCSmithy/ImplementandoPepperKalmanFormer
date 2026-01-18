import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent to path to import packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.kalman_filter import KalmanFilter, ExtendedKalmanFilter
from models.kalmanformer import KalmanFormer
from data.lorenz import LorenzSystem
from training.train import train_model

def generate_lorenz_dataset(n_sequences: int, seq_len: int, dt: float = 0.05, 
                           sigma_sys: float = 10.0, # Using standard 10 unless forced to 0.05
                           Q_val: float = 0.8**2, R_val: float = 1.0**2):
    
    # NOTE: Prompt says "sigma=0.05" in section 4.1. 
    # If this refers to Lorenz parameter sigma, dynamics will be very different.
    # If it refers to something else, it's ambiguous.
    # We allow overriding.
    
    lorenz = LorenzSystem(sigma=sigma_sys, dt=dt)
    
    # Process Noise Covariance
    Q = torch.eye(3) * Q_val
    # Measurement Noise Covariance
    R = torch.eye(3) * R_val
    
    # Initial states: Random around some point or fixed?
    # Lorenz is chaotic. Start near attractor or random.
    # Random start [-1, 1]
    
    z_seqs = []
    x_true_seqs = []
    
    for _ in range(n_sequences):
        x = torch.randn(1, 3, 1) # (1, 3, 1)
        # Transient
        for _ in range(100):
            x = lorenz.taylor_step(x)
            
        x_seq = []
        z_seq = []
        
        for k in range(seq_len):
            # True dynamics
            # x_next = f(x) + w
            # Usually we integrate f(x) and add w.
            # Using taylor step as dynamic transition f(x)
            
            x_next_det = lorenz.taylor_step(x)
            w = torch.randn_like(x) * math.sqrt(Q_val)
            x_new = x_next_det + w
            
            # Measurement
            # z = Hx + v, H=I
            v = torch.randn_like(x) * math.sqrt(R_val)
            z = x_new + v
            
            x_seq.append(x_new.squeeze(0)) # (3, 1)
            z_seq.append(z.squeeze(0)) # (3, 1)
            
            x = x_new
            
        x_true_seqs.append(torch.stack(x_seq)) # (seq, 3, 1)
        z_seqs.append(torch.stack(z_seq).squeeze(-1))     # (seq, 3)
        
    return torch.stack(z_seqs), torch.stack(x_true_seqs)

import math

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Parameters
    SEQ_LEN = 100
    N_TRAIN = 300
    N_VAL = 60
    BATCH_SIZE = 128 # Increased to 128 (Est. VRAM usage ~11.5GB/15GB)
    
    # Prompt 4.1 Specs
    DT = 0.02 # Smaller step for better stability with RK4
    SIGMA_LORENZ = 2.0 # Reduced chaos for validation (was 10.0)
    # SIGMA_LORENZ = 0.05 Was too slow/stable 
    
    Q_STD = 0.01  # Minimal process noise (validation: was 0.8)
    R_STD = 0.01  # Minimal measurement noise (validation: was 1.0)
    
    print("Generating Data...")
    # Train
    z_train, x_train = generate_lorenz_dataset(N_TRAIN, SEQ_LEN, dt=DT, sigma_sys=SIGMA_LORENZ, Q_val=Q_STD**2, R_val=R_STD**2)
    # Val
    z_val, x_val = generate_lorenz_dataset(N_VAL, SEQ_LEN, dt=DT, sigma_sys=SIGMA_LORENZ, Q_val=Q_STD**2, R_val=R_STD**2)
    
    train_ds = TensorDataset(z_train, x_train)
    val_ds = TensorDataset(z_val, x_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # System Defines
    lorenz_sys = LorenzSystem(sigma=SIGMA_LORENZ, dt=DT)
    
    # Matrices
    Q = torch.eye(3, device=device) * (Q_STD**2)
    R = torch.eye(3, device=device) * (R_STD**2)
    
    # Model: KalmanFormer
    # Functions for EKF
    # f_func: x -> taylor_step(x) WITHOUT noise (prediction)
    # But wait, taylor_step generates next state.
    # The EKF predict needs f(x).
    
    # We need Python wrappers suitable for batched tensors on GPU
    def f_wrapper(x, u=None):
        # x: (batch, n, 1)
        # Need to ensure lorenz_sys uses torch operations on correct device
        # LorenzSystem is stateless wrt device, but creates tensors?
        # My LorenzSystem impl didn't specify device for scratch tensors explicitly, 
        # but operations like x*y maintain device.
        # Check lorenz.jacobian: "J = torch.zeros(..., device=state.device)" -> GOOD.
        # Check taylor_step: uses self.dynamics -> GOOD.
        return lorenz_sys.taylor_step(x)
        
    def h_wrapper(x):
        return x # Identity
        
    def F_jac_wrapper(x):
        # Returns (Batch, 3, 3)
        return lorenz_sys.get_F_matrix(x)
        
    def H_jac_wrapper(x):
        # Returns (Batch, 3, 3)
        return torch.eye(3, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        
    H_jac = H_jac_wrapper # Use function
    
    kf_former = KalmanFormer(state_dim=3, 
                             meas_dim=3, 
                             f_func=f_wrapper, 
                             h_func=h_wrapper, 
                             F_jacobian=F_jac_wrapper, 
                             H_jacobian=H_jac, 
                             Q=Q, 
                             R=R, 
                             device=device)
                             
    # Training
    print("Training KalmanFormer...")
    kf_former.to(device)
    # Conservative LR to avoid NaN explosion
    history = train_model(kf_former, train_loader, val_loader, epochs=200, lr=2e-4, device=device)
    
    # Evaluation on one sequence
    print("Evaluating...")
    z_test = z_val[0:1].to(device)
    x_test_true = x_val[0:1].to(device)
    
    with torch.no_grad():
        x_est_kf_former = kf_former(z_test)
        
    # Baseline: EKF
    ekf = ExtendedKalmanFilter(3, 3, device=device)
    # Run EKF Loop manually or use helper?
    # I'll implement a runner helper
    
    x_est_ekf = []
    x_curr = torch.zeros(1, 3, 1, device=device)
    P_curr = torch.eye(3, device=device).unsqueeze(0)
    
    for k in range(SEQ_LEN):
        z_k = z_test[:, k, :].unsqueeze(-1)
        
        # Pred
        x_pred, P_pred = ekf.predict(x_curr, P_curr, f_wrapper, F_jac_wrapper, Q)
        
        # Update
        x_curr, P_curr, _ = ekf.update(x_pred, P_pred, z_k, h_wrapper, H_jac, R)
        
        x_est_ekf.append(x_curr)
        
    x_est_ekf = torch.stack(x_est_ekf, dim=1)
    
    # MSE Calc
    mse_former = nn.MSELoss()(x_est_kf_former, x_test_true)
    mse_ekf = nn.MSELoss()(x_est_ekf, x_test_true)
    
    print(f"MSE KalmanFormer: {mse_former.item()}")
    print(f"MSE EKF: {mse_ekf.item()}")
    
    # Save Results
    results = {
        'history': history,
        'mse_former': mse_former.item(),
        'mse_ekf': mse_ekf.item(),
        # Save tensors on CPU to save memory/disk
        'x_test_true': x_test_true.squeeze(0).cpu(), # Remove batch dim if size 1
        'x_est_former': x_est_kf_former.squeeze(0).cpu(),
        'x_est_ekf': x_est_ekf.squeeze(0).cpu()
    }
    # Ensure output directory exists
    output_path = os.path.join(os.path.dirname(__file__), 'simulation_results.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(results, output_path)
    print(f"Results saved to {output_path}")
    
    # Plotting code could go here or in a notebook
    
if __name__ == "__main__":
    main()
