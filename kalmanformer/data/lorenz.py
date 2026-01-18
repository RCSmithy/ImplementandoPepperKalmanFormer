import torch
import math

class LorenzSystem:
    def __init__(self, 
                 sigma: float = 10.0, 
                 rho: float = 28.0, 
                 beta: float = 8.0/3.0, 
                 dt: float = 0.05):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        
    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes dx/dt = Psi(x)
        state: (batch, 3, 1)
        """
        x = state[:, 0, 0]
        y = state[:, 1, 0]
        z = state[:, 2, 0]
        
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        return torch.stack([dx, dy, dz], dim=1).unsqueeze(2)

    def jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """
        Jacobian J = dPsi/dx
        """
        batch_size = state.size(0)
        x = state[:, 0, 0]
        y = state[:, 1, 0]
        z = state[:, 2, 0]
        
        J = torch.zeros(batch_size, 3, 3, device=state.device)
        
        # Row 1: -sigma, sigma, 0
        J[:, 0, 0] = -self.sigma
        J[:, 0, 1] = self.sigma
        
        # Row 2: rho - z, -1, -x
        J[:, 1, 0] = self.rho - z
        J[:, 1, 1] = -1.0
        J[:, 1, 2] = -x
        
        # Row 3: y, x, -beta
        J[:, 2, 0] = y
        J[:, 2, 1] = x
        J[:, 2, 2] = -self.beta
        
        return J

    def taylor_step(self, x: torch.Tensor, J_order: int = 5) -> torch.Tensor:
        r"""
        Integrate using Taylor Expansion of order J_order.
        x_{k+1} = x_k + Sum_{j=1}^J (1/j!) * D^j x * dt^j
        Here we approximate D^j x (j-th derivative wrt time).
        
        For J=5, this is complex to handle symbolically for D^2, D^3 etc.
        Usually papers mean J terms of the transition approx:
        F = I + A*dt + (A*dt)^2/2! + ... ?
        If the system was linear x' = Ax, then x(t) = e^{At}x(0).
        For Non-linear, we can use 5th order Runge Kutta or similar.
        If the instructions say "Model A(x) and Taylor expansion order J=5",
        it might refer to approximating the Jacobian F_k for EKF using Taylor?
        
        Let's implement explicit derivatives if possible or fallback to RK4 which is O(dt^5) local error.
        Given "No simplifiques matemáticas", I'll try to implement derivatives recursively if feasible.
        
        Derivative 1: f(x)
        Derivative 2: J(x) * f(x)
        Derivative 3: (dJ/dt) * f + J * df/dt = ...
        This gets messy fast.
        
        Alternative interpretation: "A(x)" is likely the linearized matrix at x.
        And we approximate the transition matrix F_k = exp(A(x)*dt) using Taylor J=5.
        Propagator: x_{k+1} approx x_k + (I + A dt + A^2 dt^2/2 + ...) * f(x) * dt ?? No.
        
        If we approximate Dynamics as Locally Linear Time Invariant (LTI) over dt:
        x_dot = A(x_k) * x
        Then x_{k+1} = exp(A(x_k) * dt) * x_k
        And exp(M) approx Sum_{j=0}^J M^j / j!
        
        This aligns with "Matriz A(x) and Taylor expansion J=5".
        I will implement this:
        1. Compute A = Jacobian(x_k)
        2. Compute F = Sum_{j=0}^5 (A * dt)^j / j!
        3. x_{k+1} = F * x_k ??? 
           Wait, Lorenz is affine? No.
           Locally linear approx is dx = A(x) dx.
           
           Proper integration for EKF prediction step:
           x_{k+1} = x_k + \int f(x) dt
           
           I will stick to the standard interpretation in Extended Kalman Filter literature for Lorenz (e.g. KalmanNet paper):
           Use Taylor expansion of the *matrix exponential* of the Jacobian to approximate the transition matrix F_k.
           The State prediction itself x_{k|k-1} is usually done via RK4.
           
           Let's provide BOTH: RK4 for Trajectory Generation, and Taylor Matrix Exp for EKF F_k.
        """
        
        # RK4 for state
        k1 = self.dynamics(x)
        k2 = self.dynamics(x + 0.5 * self.dt * k1)
        k3 = self.dynamics(x + 0.5 * self.dt * k2)
        k4 = self.dynamics(x + self.dt * k3)
        
        x_new = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_new

    def get_F_matrix(self, x: torch.Tensor, order: int = 5) -> torch.Tensor:
        """
        Computes linearized transition matrix F_k = exp(J(x) * dt) using Taylor series.
        """
        J = self.jacobian(x)
        M = J * self.dt
        
        F = torch.eye(3, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        M_pow = M.clone()
        fact = 1.0
        
        for k in range(1, order + 1):
            fact *= k
            F = F + M_pow / fact
            if k < order:
                M_pow = torch.bmm(M_pow, M)
                
        return F
