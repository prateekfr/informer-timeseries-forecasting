# model/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import ProbSparseAttention, ScaledDotProductAttention

class ConvLayer(nn.Module):
    """
    Convolutional Distillation Layer
    (downsamples the sequence to extract key features and reduce sequence length)
    """
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)  # (B, L, D)
        return x

class EncoderLayer(nn.Module):
    """
    Single Encoder Layer:
    - Multi-head Attention (ProbSparse Attention)
    - Feed-forward Network
    - Layer Normalization
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Attention block
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed Forward Network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        return out, attn

class Encoder(nn.Module):
    """
    Stack multiple EncoderLayers and optionally apply convolutional distillation
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attentions = []
        for idx, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attentions.append(attn)

            if self.conv_layers is not None and idx < len(self.conv_layers):
                x = self.conv_layers[idx](x)

        if self.norm is not None:
            x = self.norm(x)

        return x, attentions
