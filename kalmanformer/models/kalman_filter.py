import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple, Union

class KalmanFilter(nn.Module):
    """
    Standard Recursive Kalman Filter Implementation in PyTorch.
    
    System Model:
        x_k = F_k * x_{k-1} + B_k * u_{k-1} + w_{k-1},  w ~ N(0, Q)
        z_k = H_k * x_k + v_k,                          v ~ N(0, R)
    """
    def __init__(self, state_dim: int, meas_dim: int, device: str = 'cpu'):
        super(KalmanFilter, self).__init__()
        self.n = state_dim
        self.m = meas_dim
        self.device = device
        self.I = torch.eye(self.n, device=self.device)

    def predict(self, 
                x_est: torch.Tensor, 
                P_est: torch.Tensor, 
                F: torch.Tensor, 
                Q: torch.Tensor, 
                B: Optional[torch.Tensor] = None, 
                u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Phase:
        x_{k|k-1} = F_k * x_{k-1|k-1} + B * u_{k-1}
        P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q_{k-1}
        """
        # x_pred: [batch, n, 1]
        x_pred = torch.bmm(F, x_est)
        
        if B is not None and u is not None:
            x_pred = x_pred + torch.bmm(B, u)
            
        # P_pred: [batch, n, n]
        P_pred = torch.bmm(torch.bmm(F, P_est), F.transpose(1, 2)) + Q
        
        return x_pred, P_pred

    def update(self, 
               x_pred: torch.Tensor, 
               P_pred: torch.Tensor, 
               z: torch.Tensor, 
               H: torch.Tensor, 
               R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update Phase:
        K_k = P_{k|k-1} * H_k^T / (H_k * P_{k|k-1} * H_k^T + R_k)
        x_{k|k} = x_{k|k-1} + K_k * (z_k - H_k * x_{k|k-1})
        P_{k|k} = (I - K_k * H_k) * P_{k|k-1}
        """
        # S = H P H^T + R
        H_T = H.transpose(1, 2)
        S = torch.bmm(torch.bmm(H, P_pred), H_T) + R
        
        # Add epsilon for numerical stability
        eps = 1e-6
        S = S + eps * torch.eye(S.size(1), device=S.device).unsqueeze(0)
        
        # Kalman Gain K = P H^T S^-1
        # Using torch.linalg.solve for stability: K = (S^-1 H P)^T -> not exactly, let's stick to standard definition
        # K = P * H^T * inv(S)
        S_inv = torch.inverse(S) # Or torch.linalg.inv
        K = torch.bmm(torch.bmm(P_pred, H_T), S_inv)
        
        # Innovation y = z - H x
        z_pred = torch.bmm(H, x_pred)
        y = z - z_pred
        
        # State Update
        x_new = x_pred + torch.bmm(K, y)
        
        # Covariance Update
        # P_new = (I - K H) P_pred
        # Robust form (Joseph form) could be used but standard is requested
        I_batch = self.I.unsqueeze(0).expand(x_pred.size(0), -1, -1)
        IKH = I_batch - torch.bmm(K, H)
        P_new = torch.bmm(IKH, P_pred)
        
        return x_new, P_new, K


class ExtendedKalmanFilter(nn.Module):
    """
    Extended Kalman Filter (EKF) Implementation.
    
    Model:
        x_k = f(x_{k-1}, u_{k-1}) + w_k
        z_k = h(x_k) + v_k
    """
    def __init__(self, state_dim: int, meas_dim: int, device: str = 'cpu'):
        super(ExtendedKalmanFilter, self).__init__()
        self.n = state_dim
        self.m = meas_dim
        self.device = device
        self.I = torch.eye(self.n, device=self.device)
        
    def predict(self, 
                x_est: torch.Tensor, 
                P_est: torch.Tensor, 
                f_func: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
                F_jacobian: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
                Q: torch.Tensor,
                u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EKF Predict Phase.
        F_jacobian can be a fixed Tensor or a function F(x).
        """
        # x_pred comes from non-linear function f
        x_pred = f_func(x_est, u)
        
        # Calculate dynamic Jacobian if needed
        if callable(F_jacobian):
            F = F_jacobian(x_est) # Jacobian at previous estimate
        else:
            F = F_jacobian
            
        # P_pred uses linearized system F
        # P_pred = F P F^T + Q
        # Note: BMM expects (batch, n, n).
        P_pred = torch.bmm(torch.bmm(F, P_est), F.transpose(1, 2)) + Q
        
        return x_pred, P_pred

    def update(self, 
               x_pred: torch.Tensor, 
               P_pred: torch.Tensor, 
               z: torch.Tensor, 
               h_func: Callable[[torch.Tensor], torch.Tensor],
               H_jacobian: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
               R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EKF Update Phase.
        H_jacobian can be a fixed Tensor or a function H(x).
        """
        # Measurement prediction from non-linear function h
        z_pred = h_func(x_pred)
        
        if callable(H_jacobian):
            H = H_jacobian(x_pred) # Jacobian at predicted state
        else:
            H = H_jacobian
            
        # S = H P H^T + R
        H_T = H.transpose(1, 2)

        S = torch.bmm(torch.bmm(H, P_pred), H_T) + R
        
        # Add epsilon
        eps = 1e-6
        S = S + eps * torch.eye(S.size(1), device=S.device).unsqueeze(0)
        
        # Kalman Gain
        S_inv = torch.inverse(S)
        K = torch.bmm(torch.bmm(P_pred, H_T), S_inv)
        
        # Innovation
        y = z - z_pred
        
        # Updates
        x_new = x_pred + torch.bmm(K, y)
        
        I_batch = self.I.unsqueeze(0).expand(x_pred.size(0), -1, -1)
        IKH = I_batch - torch.bmm(K, H)
        P_new = torch.bmm(IKH, P_pred)
        
        return x_new, P_new, K
