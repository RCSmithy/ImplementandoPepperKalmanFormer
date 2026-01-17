import torch
from .simulation import generate_lorenz_dataset
# Import model classes

def run_robustness_tests():
    """
    4.3 Robustez
    - Rotación 3D
    - Desajuste F
    - Desajuste H
    - Proyección esférica
    """
    # 1. Rotation 3D
    # Apply Rotation Matrix to z_seq
    
    # 2. Mismatch F
    # Init KalmanFormer with wrong F_jacobian (e.g. wrong sigma)
    
    # 3. Mismatch H
    # Init with wrong H (e.g. 0.5 * I)
    
    pass

if __name__ == "__main__":
    run_robustness_tests()
