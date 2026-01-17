import torch
import torch.nn as nn

def compute_loss(x_est_seq: torch.Tensor, x_true_seq: torch.Tensor, model: nn.Module, lambda_reg: float = 1e-4) -> torch.Tensor:
    """
    Computes (1/T) * Sum ||x_est - x_true||^2 + lambda * ||Theta||^2
    """
    # MSE
    mse = nn.MSELoss()(x_est_seq, x_true_seq)
    
    # L2 Regularization (Weight Decay)
    # Usually handled by optimizer (weight_decay), but if explicit:
    l2_reg = torch.tensor(0., device=x_est_seq.device)
    for param in model.parameters():
        l2_reg += torch.norm(param) ** 2
        
    return mse + lambda_reg * l2_reg
