# utils.py

import numpy as np
import torch
import math

def positional_encoding(seq_len, d_model):
    """
    Create standard sinusoidal positional encodings.
    Args:
        seq_len: length of the sequence (e.g., 96)
        d_model: dimension of the model (e.g., 512)
    Returns:
        pe: (seq_len, d_model) tensor of positional encodings
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)   # even indexes
    pe[:, 1::2] = np.cos(position * div_term)   # odd indexes
    
    pe = torch.tensor(pe, dtype=torch.float32)
    return pe

def create_padding_mask(seq_q, seq_k):
    """
    Create a padding mask (if needed for variable length inputs).
    Here we assume fixed-length input, so this is optional.
    """
    len_q = seq_q.size(1)
    len_k = seq_k.size(1)
    mask = torch.ones(len_q, len_k)
    return mask.bool()

def create_look_ahead_mask(seq_len):
    """
    Create a look-ahead mask to prevent attention to future positions.
    Used in decoder during training.
    """
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    return mask.bool()
