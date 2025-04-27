# replot_training_loss.py

import numpy as np
import matplotlib.pyplot as plt

# Load saved losses
losses = np.loadtxt('training_losses.txt')

# Plot the training loss
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.savefig('replotted_training_loss.png')  # Save a new plot
plt.show()
