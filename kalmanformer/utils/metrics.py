import torch

def calculate_mse(x_true, x_est):
    return torch.mean((x_true - x_est)**2).item()

def calculate_rmse(x_true, x_est):
    return torch.sqrt(torch.mean((x_true - x_est)**2)).item()

def euclidean_distance(x_true, x_est):
    # Mean Euclidean distance over sequence
    # x: (seq, dim)
    dist = torch.norm(x_true - x_est, dim=-1)
    return torch.mean(dist).item()
