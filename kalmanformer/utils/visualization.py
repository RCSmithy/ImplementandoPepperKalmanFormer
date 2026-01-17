import matplotlib.pyplot as plt
import torch

def plot_trajectory(x_true, x_est, title="Trajectory"):
    """
    x_true: (seq, 3) or (seq, n)
    x_est: (seq, 3)
    """
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.cpu().detach().numpy()
    if isinstance(x_est, torch.Tensor):
        x_est = x_est.cpu().detach().numpy()
        
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], label='True', lw=1)
    ax.plot(x_est[:, 0], x_est[:, 1], x_est[:, 2], label='Est', lw=1, linestyle='--')
    ax.set_title(title + " (3D)")
    ax.legend()
    
    # 2D Projections
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_true[:, 0], x_true[:, 1], label='True XY')
    ax2.plot(x_est[:, 0], x_est[:, 1], label='Est XY', linestyle='--')
    ax2.set_title("XY Projection")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_errors(x_true, x_est):
    """
    Plots error over time per dimension.
    """
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.cpu().detach().numpy()
    if isinstance(x_est, torch.Tensor):
        x_est = x_est.cpu().detach().numpy()
        
    err = x_true - x_est
    dims = err.shape[1]
    
    fig, axes = plt.subplots(dims, 1, figsize=(10, 2*dims))
    for i in range(dims):
        axes[i].plot(err[:, i])
        axes[i].set_title(f"Error Dim {i}")
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()
