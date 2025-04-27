# test.py

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
    print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Testing on CPU")

# Configurations (must match train.py)
seq_len = 96
label_len = 48
pred_len = 24
enc_in = 1
dec_in = 1
c_out = 1
batch_size = 1  # For forecasting, usually batch_size = 1
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

# Forecasting
predictions = []
ground_truths = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        dec_inp = torch.zeros_like(batch_y[:, :label_len, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp[:, label_len:, :]], dim=1).to(device)

        outputs = model(batch_x, dec_inp)

        predictions.append(outputs.cpu().numpy())
        ground_truths.append(batch_y[:, -pred_len:, :].cpu().numpy())

# Convert to arrays
predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)

# Flatten for plotting
predictions_flat = predictions.reshape(-1)
ground_truths_flat = ground_truths.reshape(-1)

# Save predictions
if not os.path.exists('runs/testing'):
    os.makedirs('runs/testing', exist_ok=True)

np.savetxt('runs/testing/predicted_values.txt', predictions_flat)
np.savetxt('runs/testing/true_values.txt', ground_truths_flat)

print("\nForecasted predictions saved to 'predictions/' folder.")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ground_truths_flat[:100], label='Ground Truth')
plt.plot(predictions_flat[:100], label='Forecast')
plt.legend()
plt.title('Informer Forecast vs Ground Truth')
plt.grid(True)
plt.savefig('runs/testing/forecast_plot.png')
plt.show()
