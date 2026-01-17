# Project Walkthrough: KalmanFormer Implementation

This document serves as a detailed guide to the implemented codebase, explaining the design decisions, file responsibilities, and specific functional flows.

## 1. Project Philosophy & Architecture

The project is structured to separate **System Dynamics** (the physics/math), **Estimation Models** (the AI/Filter), and **Experimentation** (running the loop).

### Core Components

#### `models/kalman_filter.py`
Contains the "Classic" approaches.
- **`KalmanFilter`**: The textbook implementation. We use `torch.bmm` (Batch Matrix Multiplication) for everything to allow parallel simulation of 30+ trajectories on GPU.
- **`ExtendedKalmanFilter`**: The baseline for non-linear systems.
  - **Key Design Choice**: The `predict` and `update` methods accept `F_jacobian` and `H_jacobian` as either static Tensors or Callables. This allows handling both simple systems (Constant Jacobian) and complex ones (State-dependent Jacobian) without changing the class code.

#### `models/transformer.py`
A "Glass Box" implementation of the Transformer.
- We did not use `torch.nn.TransformerEncoder`. Why? To ensure exact adherence to the prompt's mathematical description (e.g., the specific embedding dimension rules and positional encoding formulas).
- **`PositionalEncoding`**: Implements the $sin(k/n...)$ formula explicitly.
- **`TransformerModel`**: A custom Encoder-Decoder wrapper that handles the separate feature implementations for state and measurement histories.

#### `models/kalmanformer.py`
The "Brain" of the operation.
- **Hybrid FLow**: It doesn't just predict a value; it intervenes in the Kalman Loop.
- **Stateful Recurrence**: Unlike a standard Transformer that processes a whole sequence at once for translation, estimation is temporal. However, we implemented a loop that runs step-by-step to feed the *result* of step $k$ (the estimate $\hat{x}_k$) back into the *input* features for step $k+1$. This makes training slower than a pure parallel Transformer but is mathematically necessary for closed-loop estimation.

---

## 2. Data Generation: The Lorenz System

Located in `data/lorenz.py`.

The Lorenz System is a chaotic ODE.
$$
\frac{dx}{dt} = \sigma(y - x)
$$
To use this in a discrete EKF, we need:
1.  **State Integration**: We used **RK4 (Runge-Kutta 4)**. This is much more stable than Euler integration for the step size $\Delta t = 0.05$.
2.  **Transition Matrix**: The EKF needs a matrix $F_k$ such that $x_{k+1} \approx F_k x_k$. Since Lorenz is non-linear, $F_k$ is the matrix exponential of the Jacobian. We implemented a **Taylor Expansion (Order 5)** to approximate this exponential, providing a highly accurate locally-linear model for the EKF baseline.

---

## 3. Experimentation Flow

### Simulation (`experiments/simulation.py`)

This script is the main entry point.
1.  **Data Gen**: Calls `generate_lorenz_dataset`. It creates `N=300` trajectories.
2.  **Model Setup**: Initializes `KalmanFormer` with specific dimensions.
3.  **Training**:
    - Optimizer: Adam
    - Scheduler: Cosine Annealing (starts high, drops smoothly).
    - Loss: MSE + Regularization.
4.  **Evaluation**:
    - Takes a held-out test trajectory.
    - Runs `KalmanFormer`.
    - Runs standard `EKF`.
    - Compares MSE.

**How to Read Results**:
The script saves `experiments/simulation_results.pt`. This dictionary contains:
- Training Loss History (list)
- Validation Loss History (list)
- Final Test MSEs for both models.

---

## 4. Visual Verification

We included `utils/visualization.py` to plot the chaotic attractors.
- **`plot_trajectory`**: Plots the "Butterfly" attractor in 3D. If the estimator is working, the "Est" dashed line should tightly hug the "True" solid line, even as it spirals around the two lobes.

---

## 5. Robustness & Multi-Sensor (Next Steps)

Skeletons are provided in `experiments/multisensor.py` and `robustness.py`.
- **Robustness**: The logical next step is to introduce "Mismatch". In the current simulation, the EKF knows the exact equation parameters ($\sigma, \rho, \beta$). A true test of KalmanFormer is to give the EKF wrong parameters (e.g., $\sigma=12.0$) while the KalmanFormer learns to compensate for this model error from data.

---

**End of Walkthrough**
