import torch
import matplotlib.pyplot as plt
import os
import sys

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load results
results_path = os.path.join(os.path.dirname(__file__), '../experiments/simulation_results.pt')

print(f"Cargando {results_path}...")
results = torch.load(results_path, map_location='cpu')

if 'history' not in results:
    print("❌ No se encontró 'history' en el archivo de resultados.")
    exit(1)

history = results['history']
train_loss = history['train_loss']
val_loss = history['val_loss']

print(f"✅ Historia cargada: {len(train_loss)} épocas")

# Plot training curves
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

epochs = range(1, len(train_loss) + 1)
ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Curva de Aprendizaje del KalmanFormer', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add final values as text
final_train = train_loss[-1]
final_val = val_loss[-1]
ax.text(0.95, 0.95, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Check convergence
if len(train_loss) > 10:
    improvement = train_loss[0] - train_loss[-1]
    if improvement < 0.01:
        ax.text(0.5, 0.5, '⚠️ MODELO NO CONVERGIÓ', transform=ax.transAxes,
                fontsize=20, color='red', alpha=0.3, ha='center', va='center',
                rotation=30)

plt.tight_layout()
print("Mostrando gráfica...")
plt.show()

# También imprimir métricas finales
print("\n" + "="*50)
print("MÉTRICAS FINALES")
print("="*50)
print(f"MSE KalmanFormer: {results.get('mse_former', 'N/A')}")
print(f"MSE EKF: {results.get('mse_ekf', 'N/A')}")
print(f"Train Loss (última época): {final_train:.6f}")
print(f"Val Loss (última época): {final_val:.6f}")
print(f"Mejora total (Loss): {train_loss[0]:.4f} → {final_train:.4f}")
print("="*50)
