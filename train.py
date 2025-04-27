# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from model.informer import Informer
from data_loader import TimeSeriesDataset

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("Training on CPU")

# Fix random seeds for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# Configurations
seq_len = 96
label_len = 48
pred_len = 24
enc_in = 1
dec_in = 1
c_out = 1
batch_size = 32
learning_rate = 0.001
epochs = 50
d_model = 512

# Load Dataset
train_dataset = TimeSeriesDataset(
    csv_file='data/ETT-small.csv',  
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    features=['OT'],
    target='OT'
)

# Split into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
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
    e_layers=3,  # Increased encoder layers
    d_layers=1,
    d_ff=2048,
    dropout=0.05,
    attn='prob',
    activation='gelu',
    output_attention=False
).to(device)

# Loss, optimizer and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Folder setup
os.makedirs('runs/training/checkpoints', exist_ok=True)

# Training loop
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    tq = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_x, batch_y in tq:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        dec_inp = torch.zeros_like(batch_y[:, :label_len, :]).float()
        dec_inp = torch.zeros(batch_y.shape[0], label_len + pred_len, batch_y.shape[2]).float().to(device)
        dec_inp[:, :label_len, :] = batch_y[:, :label_len, :]



        optimizer.zero_grad()

        outputs = model(batch_x, dec_inp)
        loss = criterion(outputs, batch_y[:, -pred_len:, :])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        tq.set_postfix({'batch_loss': loss.item()})

    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)

    print(f"\nEpoch {epoch+1}/{epochs} - Training Loss: {avg_epoch_loss:.6f}")

    # Validation Loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            dec_inp = torch.zeros_like(batch_y[:, :label_len, :]).float()
            dec_inp = torch.zeros(batch_y.shape[0], label_len + pred_len, batch_y.shape[2]).float().to(device)
            dec_inp[:, :label_len, :] = batch_y[:, :label_len, :]



            outputs = model(batch_x, dec_inp)
            loss = criterion(outputs, batch_y[:, -pred_len:, :])
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.6f}")

    model.train()

    # Step the learning rate scheduler
    scheduler.step()

# Save model checkpoint
torch.save(model.state_dict(), 'runs/training/checkpoints/informer_checkpoint.pth')

# Save training losses
np.savetxt('runs/training/training_losses.txt', losses)

# Plot training loss
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('runs/training/training_loss_plot.png')
plt.show()
