import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class NCLTDataset(Dataset):
    """
    Loader for NCLT Dataset.
    Assumes pre-processed CSVs with columns:
    t, x, y, vx, vy, theta, omega, z_x, z_y, ... (measurements)
    """
    def __init__(self, processed_file_path: str, seq_len: int = 100):
        self.seq_len = seq_len
        try:
            self.data = pd.read_csv(processed_file_path)
            # Normalize or parse
        except FileNotFoundError:
            print(f"Warning: {processed_file_path} not found. Using dummy data.")
            self.data = pd.DataFrame(np.zeros((1000, 10)), columns=['x','y','vx','vy','theta','omega','z1','z2','z3','z4'])
            
        self.states = self.data[['x','y','vx','vy','theta','omega']].values
        self.measurements = self.data.iloc[:, 6:].values
        
    def __len__(self):
        return (len(self.data) - self.seq_len) // 10 # Stride
        
    def __getitem__(self, idx):
        # Return Sequence
        start = idx * 10
        end = start + self.seq_len
        
        x_seq = getattr(self, 'states')[start:end]
        z_seq = getattr(self, 'measurements')[start:end]
        
        return torch.tensor(z_seq, dtype=torch.float32), torch.tensor(x_seq, dtype=torch.float32).unsqueeze(-1)
