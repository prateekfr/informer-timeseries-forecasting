# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from model.informer import Informer
from data_loader import TimeSeriesDataset

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(f"Evaluating on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Evaluating on CPU")

# Configurations (should match train.py)
seq_len = 96
label_len = 48
pred_len = 24
enc_in = 1
dec_in = 1
c_out = 1
batch_size = 32
d_model = 512

# Load Dataset
test_dataset = TimeSeriesDataset(
    csv_file='data/ETT-small.csv',
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    features=['OT'],
    target='OT'
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Model
model = Informer(
    enc_in=enc_in,
    dec_in=dec_in,
    c_out=c_out,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    factor=5,
    d_model=d_model,
    n_heads=8,
    e_layers=3,
    d_layers=1,
    d_ff=2048,
    dropout=0.05,
    attn='prob',
    activation='gelu',
    output_attention=False
).to(device)

model.load_state_dict(torch.load('runs/training/checkpoints/informer_checkpoint.pth'))
model.eval()

# Evaluation metrics
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

true_vals = []
pred_vals = []

# Disable gradient calculation
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        dec_inp = torch.zeros_like(batch_y[:, :label_len, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp[:, label_len:, :]], dim=1).to(device)

        outputs = model(batch_x, dec_inp)

        true_vals.append(batch_y[:, -pred_len:, :].cpu().numpy())
        pred_vals.append(outputs.cpu().numpy())

# Concatenate all batches
true_vals = np.concatenate(true_vals, axis=0)
pred_vals = np.concatenate(pred_vals, axis=0)

# Flatten for metrics
true_vals_flat = true_vals.reshape(-1)
pred_vals_flat = pred_vals.reshape(-1)

# Compute Metrics
mae_score = MAE(true_vals_flat, pred_vals_flat)
rmse_score = RMSE(true_vals_flat, pred_vals_flat)

print(f"Test MAE: {mae_score:.6f}")
print(f"Test RMSE: {rmse_score:.6f}")

# Save evaluation results
if not os.path.exists('runs/evaluation'):
    os.makedirs('runs/evaluation', exist_ok=True)

with open('runs/evaluation/evaluation_results.txt', 'w') as f:
    f.write(f"Test MAE: {mae_score:.6f}\n")
    f.write(f"Test RMSE: {rmse_score:.6f}\n")

print("\nEvaluation results saved to 'runs/evaluation/evaluation_results.txt'")

# Plot sample predictions
plt.figure(figsize=(10, 6))
plt.plot(true_vals_flat[:100], label='Ground Truth')
plt.plot(pred_vals_flat[:100], label='Prediction')
plt.legend()
plt.title('Informer Predictions vs Ground Truth')
plt.grid(True)
plt.savefig('runs/evaluation/sample_predictions.png')
plt.show()
