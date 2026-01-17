
import sys
import os
import torch
import matplotlib.pyplot as plt

# --- 1. ARREGLO DE RUTAS ---
# Detectamos la raíz del proyecto subiendo 2 niveles desde "notebooks"
current_dir = os.getcwd()
# Assuming we are running this from VSCode interactive window or similar where cwd might be the root or the notebook dir
# We'll try to find 'kalmanformer' folder upwards
if 'kalmanformer' not in os.listdir(current_dir):
    # Maybe we are inside kalmanformer or deeper
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
else:
    project_root = current_dir

# Añadimos la raíz al sistema para que Python encuentre "kalmanformer"
if project_root not in sys.path:
    sys.path.append(project_root)

# Ahora sí podemos importar
try:
    from kalmanformer.utils.visualization import plot_trajectory, plot_errors
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print(f"Project Root detectado: {project_root}")
    print(f"Sys Path: {sys.path}")

# --- 2. CARGAR DATOS ---
results_path = os.path.join(project_root, 'kalmanformer/experiments/simulation_results.pt')

if not os.path.exists(results_path):
    print(f"❌ No encuentro el archivo: {results_path}")
    print("Ejecuta 'python kalmanformer/experiments/simulation.py' primero.")
else:
    print(f"Cargando {results_path}...")
    data = torch.load(results_path)
    print("✅ Datos cargados correctamente. Llaves:", data.keys())

    # --- 3. VISUALIZAR ---
    
    # Check what keys we actually have
    if 'x_test_true' in data:
        x_true = data['x_test_true']    # La verdad (Simulación física)
        x_est_former = data['x_est_former']    # La estimación (KalmanFormer)
        x_est_ekf = data['x_est_ekf'] # La estimación (EKF)
        
        print(f"Shapes -> True: {x_true.shape}, Former: {x_est_former.shape}, EKF: {x_est_ekf.shape}")
        
        # Eliminar dimensiones extra si existen (num_sequences, seq_len, dim) -> (seq_len, dim)
        if x_true.dim() == 3: x_true = x_true[0]
        if x_est_former.dim() == 3: x_est_former = x_est_former[0]
        if x_est_ekf.dim() == 3: x_est_ekf = x_est_ekf[0]

        # Plot KalmanFormer
        plot_trajectory(x_true, x_est_former, title="KalmanFormer Performance")
        
        # Plot EKF comparison (Optional custom plot)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_true[:, 0], label='True X')
        ax.plot(x_est_former[:, 0], label='KF-Former X', linestyle='--')
        ax.plot(x_est_ekf[:, 0], label='EKF X', linestyle=':',  alpha=0.7)
        ax.set_title("X Dimension Comparison")
        ax.legend()
        plt.show()

    else:
        print("⚠️ El archivo existe, pero NO tiene trayectorias guardadas. ¿Seguro que re-ejecutaste simulation.py?")
