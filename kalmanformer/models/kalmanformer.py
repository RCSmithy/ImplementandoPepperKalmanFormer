import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .kalman_filter import ExtendedKalmanFilter
from .transformer import TransformerModel

class KalmanFormer(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 meas_dim: int, 
                 f_func, 
                 h_func, 
                 F_jacobian, 
                 H_jacobian, 
                 Q: torch.Tensor, 
                 R: torch.Tensor,
                 device: str = 'cpu'):
        super().__init__()
        self.n = state_dim
        self.m = meas_dim
        self.device = device
        
        # System Dynamics
        self.f_func = f_func
        self.h_func = h_func
        self.F_jac = F_jacobian
        self.H_jac = H_jacobian
        self.Q = Q
        self.R = R
        
        self.ekf = ExtendedKalmanFilter(state_dim, meas_dim, device)
        
        # Transformer Parameters
        # Encoder Input: 2n (Features 3 + 4: delta_x_tilde, delta_x_hat)
        # Decoder Input: 2m (Features 1 + 2: Delta_z, delta_z)
        self.enc_input_dim = 2 * state_dim
        self.dec_input_dim = 2 * meas_dim
        
        self.transformer = TransformerModel(
            input_dim_enc=self.enc_input_dim,
            input_dim_dec=self.dec_input_dim,
            d_model=64, # As suggested by "Feed-forward dimension" being 64? Or independent? Using 64.
            num_heads=2,
            num_layers=2,
            d_ff=64
        )
        
        # Input Normalization
        self.enc_norm = nn.LayerNorm(self.enc_input_dim)
        self.dec_norm = nn.LayerNorm(self.dec_input_dim)
        
        # Output Projection to Kalman Gain K (n x m)
        self.gain_proj = nn.Linear(64, state_dim * meas_dim)
        
        # Conservative initialization to prevent early explosion
        nn.init.normal_(self.gain_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gain_proj.bias, 0.0)
        
        # Sliding window size for transformer inputs
        self.window_size = 20  # Keep last 20 steps instead of all history
        
    def _compute_features(self, 
                          z_k: torch.Tensor, 
                          z_prev: torch.Tensor, 
                          z_pred: torch.Tensor, 
                          x_hat_k: torch.Tensor,    # Current estimate (from update) - NOT AVAILABLE for Pre-Update
                          x_hat_prev: torch.Tensor, # Previous estimate
                          x_pred_k: torch.Tensor    # Predicted estimate
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes features based on Section 2.1
        NOTE: This seems circular if we need x_hat_k to predict K_k to get x_hat_k.
        Assumption:
        Encoder features (State diffs) use PAST history (up to k-1).
        Decoder features (Meas diffs) use CURRENT history (up to k).
        
        We will compute:
        Dec Feat k: Delta_z_k, delta_z_k (innovation)
        Enc Feat k-1: Delta_x_tilde_{k-1}, Delta_x_hat_{k-1}
        """
        # Decoder Features (Meas based)
        # Delta z_k = z_k - z_{k-1}
        Delta_z = z_k - z_prev
        
        # delta z_k = z_k - z_{predicted}
        delta_z = z_k - z_pred
        
        dec_feat = torch.cat([Delta_z, delta_z], dim=-1)
        
        # Encoder Features (State based) - To be stored for NEXT step's encoder input
        # computed AFTER update
        if x_hat_k is not None:
            # Delta x_tilde_k = x_hat_k - x_hat_prev
            Delta_x_tilde = x_hat_k - x_hat_prev
            
            # Delta x_hat_k = x_hat_k - x_pred_k
            Delta_x_hat = x_hat_k - x_pred_k
            
            enc_feat = torch.cat([Delta_x_tilde, Delta_x_hat], dim=-1)
        else:
            enc_feat = torch.zeros(z_k.size(0), self.enc_input_dim, device=self.device)
            
        return dec_feat, enc_feat

    def forward(self, z_seq: torch.Tensor, u_seq: Optional[torch.Tensor] = None, x_init: Optional[torch.Tensor] = None, P_init: Optional[torch.Tensor] = None):
        """
        Process a sequence.
        z_seq: (batch, seq_len, m)
        u_seq: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = z_seq.shape
        
        if x_init is None:
            x_est = torch.zeros(batch_size, self.n, 1, device=self.device)
        else:
            x_est = x_init
            
        if P_init is None:
            P_est = torch.eye(self.n, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            P_est = P_init
            
        # History buffers
        # We need to feed features to Transformer.
        # Encoder Inputs: History of state features [0...k-1]
        # Decoder Inputs: History of meas features [0...k]
        
        enc_history = [] # List of tensors (batch, 1, 2n)
        dec_history = [] # List of tensors (batch, 1, 2m)
        
        # Init Previous values
        z_prev = torch.zeros_like(z_seq[:, 0, :]).unsqueeze(-1)
        x_est_prev = x_est.clone()
        
        # Feature initialization (padding for first step)
        dummy_enc = torch.zeros(batch_size, 1, self.enc_input_dim, device=self.device)
        # enc_history.append(dummy_enc) 
        # Actually, Encoder input at step k should probably be sequence 0..k-1.
        # For k=0, input is empty? Or just dummy? 
        # Transformer needs interaction. Let's start with empty/dummy.
        
        estimates = []
        
        for k in range(seq_len):
            u_k = u_seq[:, k, :].unsqueeze(-1) if u_seq is not None else None
            z_k = z_seq[:, k, :].unsqueeze(-1)
            
            # 1. Prediction
            x_pred, P_pred = self.ekf.predict(x_est, P_est, self.f_func, self.F_jac, self.Q, u_k)
            
            # 2. Measurement Prediction (for feature)
            z_pred_val = self.h_func(x_pred)
            
            # 3. Compute Decoder Feature for Step k
            Delta_z = z_k - z_prev
            delta_z = z_k - z_pred_val
            dec_feat_k = torch.cat([Delta_z.squeeze(-1), delta_z.squeeze(-1)], dim=-1) # (B, 2m)
            dec_feat_k = self.dec_norm(dec_feat_k).unsqueeze(1) # Normalize -> (B, 1, 2m)
            dec_history.append(dec_feat_k)
            
            # 4. Prepare Transformer Inputs (with sliding window)
            # Encoder Seq: Previous state features (last window_size steps)
            # Decoder Seq: Current meas features (last window_size steps)
            
            if len(enc_history) == 0:
                 # Initial State
                 src = torch.zeros(batch_size, 1, self.enc_input_dim, device=self.device)
            else:
                 # Use sliding window instead of full history
                 start_idx = max(0, len(enc_history) - self.window_size)
                 src = torch.cat(enc_history[start_idx:], dim=1) # (B, seq_enc, 2n)
            
            # Same for decoder
            start_idx_dec = max(0, len(dec_history) - self.window_size)
            tgt = torch.cat(dec_history[start_idx_dec:], dim=1) # (B, seq_dec, 2m)
            
            # Run Transformer
            # We want K_k corresponding to the last step
            out = self.transformer(src, tgt) # (B, seq_dec, d_model)
            
            # Project to Gain (no activation - let gradient clipping handle stability)
            # Take last output
            last_out = out[:, -1, :] # (B, d_model)
            K_k_flat = self.gain_proj(last_out)
            K_k = K_k_flat.view(batch_size, self.n, self.m)
            
            # 5. Hybrid Update
            # x_new = x_pred + K_k * (z_k - H * x_pred)
            # Note: Prompt says "z_k - H_k * x_pred". 
            # In EKF, innovation is z_k - h(x_pred). 
            # If using K_k from Transformer, we use that K_k instead of optimal K.
            
            # Innovation
            y = z_k - z_pred_val # using non-linear h if diff is h(x)
            
            x_new = x_pred + torch.bmm(K_k, y)
            
            # P Update? 
            # Hybrid usually only learns State, P might become inconsistent.
            # But we need P for next prediction step in EKF.
            # Using Joseph form with Learned K:
            # P = (I - K H) P (I - K H)^T + K R K^T
            # Or just standard: P = (I - KH) P_pred
            # Prompt doesn't specify P update for Hybrid, assume standard using predicted K.
            
            H_val = self.H_jac # Approximation if H is constant, else needs re-eval?
            # EKF H is Jacobian. We'll use H_jac passed in init? 
            # Wait, EKF needs H_k computed at current step usually.
            # I should use passed H_jacobian if constant or ...
            # The prompt says H_k = partial h / partial x.
            # Ideally compute Jacobian at x_pred.
            # If H_jacobian is a Tensor, it's constant. If it's a func?
            # Init args say 'H_jacobian'. I'll assume it's a fixed tensor for Lorenz exp?
            # Actually for Lorenz, H=I (linear).
            # For NCLT, it might be non-linear.
            # I will assume F_jacobian and H_jacobian in __init__ are TENSORS (Constant approximation) or I should allowed them to be calculated.
            # For flexibility, I'll stick to using the passed tensors for now, assuming constant approx or pre-computed.
            # BUT, standard EKF re-computes Jacobians.
            # Let's rely on the user passing 'F_jacobian' as a value or Update logic?
            # "F_k = partial f ...". This changes.
            # My 'EKF' class takes 'F_jacobian' as arg in 'predict'.
            # Here in 'KalmanFormer', I pass self.F_jac. 
            # If the system is non-linear, I should update F_jac.
            # I will leave it fixed for now or assume passed values are sufficient for the 'Experiment 4.1' where A(x) is Taylor expansion.
            
            # Let's finish the loop first.
            # Evaluate H if callable
            if callable(self.H_jac):
                H = self.H_jac(x_pred)
            else:
                H = self.H_jac
                
            I_batch = torch.eye(self.n, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
            IKH = I_batch - torch.bmm(K_k, H)
            P_new = torch.bmm(IKH, P_pred) # Simplified update
            
            estimates.append(x_new)
            
            # 6. Compute Encoder Features for NEXT step
            # Delta_x_tilde = x_new - x_est_prev
            # Delta_x_hat = x_new - x_pred
            Delta_x_tilde = x_new - x_est_prev
            Delta_x_hat = x_new - x_pred
            
            enc_feat_k = torch.cat([Delta_x_tilde.squeeze(-1), Delta_x_hat.squeeze(-1)], dim=-1) # (B, 2n)
            enc_feat_k = self.enc_norm(enc_feat_k).unsqueeze(1) # Normalize -> (B, 1, 2n)
            enc_history.append(enc_feat_k)
            
            # Update States
            x_est = x_new
            x_est_prev = x_new
            P_est = P_new
            z_prev = z_k
            
        return torch.stack(estimates, dim=1)


# ## Resumen del Flujo Completo
# ```
# Para cada paso k:
#   1. EKF predice: x_pred, P_pred
#   2. Calcula características: Delta_z, delta_z (decoder)
#   3. Alimenta Transformer con:
#      - Encoder: historia de características de estado [0...k-1]
#      - Decoder: historia de características de medición [0...k]
#   4. Transformer → K_k (ganancia aprendida)
#   5. Actualiza: x_new = x_pred + K_k * (z_k - z_pred)
#   6. Actualiza: P_new usando K_k
#   7. Calcula características de estado: Delta_x_tilde, Delta_x_hat
#   8. Añade a historia para siguiente paso