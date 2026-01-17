# KalmanFormer: Deep Learning for Dynamic State Estimation

## Overview
**KalmanFormer** is a hybrid state estimation framework that combines the theoretical robustness of **Kalman Filters (KF/EKF)** with the representational power of **Transformers**. This repository contains a PyTorch implementation of KalmanFormer, designed to be scientifically reproducible and modular.

The system uses a Transformer to learn the **Kalman Gain ($K_k$)** directly from data features (state and measurement innovations), replacing the analytical gain computation in complex, non-linear, or mathematically intractable scenarios.

## 📁 Project Structure

```
kalmanformer/
├── models/
│   ├── kalman_filter.py     # Base KF and EKF implementations (PyTorch)
│   ├── transformer.py       # Custom Transformer Encoder/Decoder with Multi-Head Attention
│   └── kalmanformer.py      # Hybrid Architecture (The Core Model)
├── data/
│   ├── lorenz.py            # Lorenz System dynamics, RK4 integration, Taylor Expansion Linearization
│   └── nclt_loader.py       # Dataset loader for NCLT (skeleton)
├── training/
│   ├── loss.py              # Custom Loss: MSE + Regularization
│   └── train.py             # Std Training Loop with Cosine Annealing
├── experiments/
│   ├── simulation.py        # Main Experiment: Lorenz System (Training & Evaluation)
│   ├── multisensor.py       # Multi-sensor Fusion Experiment (Stub)
│   └── robustness.py        # Robustness Tests (Stub)
└── utils/
    ├── metrics.py           # MSE, RMSE, Euclidean Distance
    └── visualization.py     # 3D Trajectory plotting
```

## 🔧 Installation & Requirements

### Prerequisites
- Python 3.8+
- PyTorch (CUDA supported)
- NumPy, Pandas, Matplotlib

### Setup
```bash
# Clone the repository
git clone <repository_url>
cd KalmanFormerAntigraviry

# Install dependencies
pip install torch numpy pandas matplotlib
```

## 🚀 Usage

### 1. Lorenz System Simulation (Quick Start)
To verify the system, run the Lorenz simulation experiment. This script generates synthetic data, trains the KalmanFormer model, and compares it against a standard Extended Kalman Filter (EKF).

```bash
python kalmanformer/experiments/simulation.py
```

**What happens:**
- Generates 300 training sequences and 60 validation sequences of Lorentz Dynamics.
- Trains `KalmanFormer` for 200 epochs.
- Saves results to `experiments/simulation_results.pt`.
- Prints MSE comparison between KalmanFormer and EKF.

### 2. Visualization
After running the simulation (locally or on Colab), you can visualize the trajectories and errors.
(Note: You can use the provided utils or the notebook `notebooks/lorenz_results.ipynb`).

### 3. Google Colab (Recommended for Fast Training)
If you don't have a powerful GPU locally, use Google Colab:
1. Upload `kalmanformer_colab.ipynb` to your Google Drive or open it directly from GitHub if supported.
2. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU** (or better).
3. Run the cells in order. The notebook will:
   - Clone this repository.
   - Install dependencies.
   - Run the full simulation and training (500 epochs).
   - Automatically provide a download link for `simulation_results.pt`.
4. Download the results file and place it in `kalmanformer/experiments/` on your local machine to visualize it.

```python
from kalmanformer.utils.visualization import plot_trajectory
import torch

results = torch.load('experiments/simulation_results.pt')
# Load your test data...
# plot_trajectory(x_true, x_est)
```

## 🧠 Theoretical Background

### Kalman Filter (KF)
Optimal estimator for linear Gaussian systems.
- **Predict**: $\hat{x}_{k|k-1} = F \hat{x}_{k-1} + B u$
- **Update**: $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})$

### Extended Kalman Filter (EKF)
For non-linear systems $x_k = f(x_{k-1}) + w$.
- Approximates linearity via Jacobians $F_k = \frac{\partial f}{\partial x}$.
- We implement this using Taylor Expansion (Order J=5) for the Lorenz system transition matrix.

### KalmanFormer (Hybrid)
Retains the predict/update structure of the KF but replaces the analytical calculation of $K_k$ (which depends on accurate $P_k$, $Q$, $R$) with a learned approach.

**Features:**
- **Decoder Input**: $\Delta z_k$ (Observation diff), $\delta z_k$ (Innovation).
- **Encoder Input**: $\Delta \tilde{x}_k$ (Update correction), $\Delta \hat{x}_k$ (Prediction step).
- **Architecture**: 2-layer Transformer, 2 Heads, 64 hidden dim.
- **Output**: Learned Kalman Gain $K_k(\Theta)$.

## 🔬 Reproducibility
- **Seeds**: Random seeds should be set in entry scripts (not fully hardcoded in library to allow variability, but `simulation.py` typically uses consistent generation).
- **Precision**: All tensors are Float32 by default.
- **Math**: Implementations (e.g., `models/transformer.py`) use explicit formulas (e.g., manual Multi-Head Attention) rather than black-box `nn.Transformer` to ensure adherence to the specific paper architecture.

## 📝 Citation
If you use this code for academic work, please cite the original KalmanFormer paper.
