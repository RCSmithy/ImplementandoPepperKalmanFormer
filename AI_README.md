# AI Context & Reproducibility Guide (AI_README.md)

**Target Audience:** Artificial Intelligence Agents, Automated Systems, and Researchers trying to reproduce this codebase from scratch or maintain it.

**Objective:** Provide absolute clarity on implementation details, mathematical assumptions, and file inter-dependencies to ensure zero ambiguity in reproduction.

---

## 1. Mathematical Implementation Details

### 1.1 System Dynamics (Lorenz)
- **Equation**: Standard Lorenz 63.
  - $\dot{x} = \sigma(y - x)$
  - $\dot{y} = x(\rho - z) - y$
  - $\dot{z} = xy - \beta z$
- **Parameters**: $\sigma=10.0, \rho=28.0, \beta=8/3$.
- **Integration**: `RK4` (Runge-Kutta 4th Order) is used for ground truth trajectory generation (`lorenz.dynamics`).
- **EKF Linearization**:
  - The EKF prediction step requires a discrete transition matrix $F_k$.
  - We approximate $F_k \approx \exp(J(x) \Delta t)$ using a **Taylor Expansion of Order 5**.
  - Implementation: `kalmanformer/data/lorenz.py -> get_F_matrix`. This is CRITICAL for EKF stability in this chaotic system. Agents modifying this must ensure $|F_k| \approx 1$ is maintained to avoid explosion.

### 1.2 Transformer Architecture
The Transformer is implemented *from scratch* (`models/transformer.py`) to match specific paper requirements, not using `torch.nn.Transformer`.
- **Positional Encoding**: Standard Sinusoidal. $P(k, 2i) = \sin(k/10000^{2i/d})$. Note: $n$ in prompt assumed 10000.
- **Attention**: Scaled Dot Product. Attention$(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
- **Feed Forward**: Linear -> ReLU -> Linear.
- **Normalization**: Pre-Norm or Post-Norm? Implementation uses `LayerNorm` *after* the residual connection (Post-Norm standard).

### 1.3 KalmanFormer Hybrid Loop
File: `models/kalmanformer.py`
- The loop is strictly sequential because $K_k$ depends on current measurement features AND past state features.
- **Dependency Graph per Step $k$**:
  1. **EKF Prediction**: $x_{k|k-1}, P_{k|k-1}$ using $f(x_{k-1})$ and $F_{k-1}$.
  2. **Feature Calc (Decoder)**: $\Delta z_k, \delta z_k$ (Observation Innovation).
  3. **Transformer Forward**: Inputs [History Enc, History Dec] -> Output $K_k$.
  4. **Hybrid Update**: $x_{k|k} = x_{k|k-1} + K_k \cdot (z_k - h(x_{k|k-1}))$.
  5. **Covariance Approximation**: $P_{k|k} = (I - K_k H) P_{k|k-1}$. (simplified Joseph form).
  6. **Feature Calc (Encoder)**: $\Delta \tilde{x}_k, \Delta \hat{x}_k$ (State Increments). Stored for step $k+1$.

---

## 2. Codebase Constraints & Invariants

1.  **Dimensions**:
    - System State ($n$): 3 (Lorenz).
    - Measurement ($m$): 3 (Full observation).
    - Transformer Dims: $d_{model}=64, d_{ff}=64, heads=2$.

2.  **Tensors**:
    - All inputs to models must be `(Batch, Sequence, Dim)`.
    - Internal EKF operations use `torch.bmm` so inputs must be 3D `(Batch, n, 1)` or `(Batch, n, n)`.
    - **CAUTION**: When stacking sequences in `simulation.py`, `z_seq` must be `(Batch, Seq, m)`. The simulation loader originally produced `(Batch, Seq, m, 1)`, requiring a `.squeeze(-1)` fix which is now applied.

3.  **Typos/Common Errors**:
    - `H_jacobian` in EKF can be a Tensor or Callable. The implementation uses `Union` typing.
    - If `H_jacobian` is a fixed Identity tensor, it MUST be expanded to `(Batch, m, n)` before use in `bmm`. The `experiments/simulation.py` handles this with `H_jac_wrapper`.

---

## 3. Reproduction Steps (Automated)

To verify the codebase functionality, an AI agent should execute the following sequence:

1.  **Environment Check**: Ensure `torch` is installed.
2.  **File Verification**: Check existence of `kalmanformer/models/kalman_filter.py`, `transformer.py`, `kalmanformer.py`.
3.  **Execution**:
    ```python
    import sys
    import os
    # Ensure root is in path
    sys.path.append(os.getcwd())
    from kalmanformer.experiments import simulation
    simulation.main()
    ```
4.  **Success Criteria**:
    - Script runs to completion without `RuntimeError`.
    - "MSE KalmanFormer" < "MSE EKF" (Usually true after sufficient training, though chaotic systems are hard).
    - `simulation_results.pt` is created.

---

## 4. Troubleshooting Guide

- **Error**: `ValueError: too many values to unpack` in `kalmanformer.forward`.
  - **Cause**: Input `z_seq` has shape `(B, L, m, 1)` instead of `(B, L, m)`.
  - **Fix**: Squeeze the last dimension of measurement inputs.

- **Error**: `RuntimeError: Expected size... [Batch, 3] but got [1, 3]`.
  - **Cause**: Broadcasting failure in `torch.bmm` or matrix ops when mixing Batched tensors with Single-instance tensors (like a fixed H matrix).
  - **Fix**: Use `.expand(batch_size, -1, -1)` on constant matrices before ops.

---

**End of AI Context**
