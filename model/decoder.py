# model/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import ProbSparseAttention, ScaledDotProductAttention

class DecoderLayer(nn.Module):
    """
    Single Decoder Layer:
    - Self-Attention (decoder attending to previous outputs)
    - Cross-Attention (decoder attending to encoder output)
    - Feed-forward network
    - Layer Normalization
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-Attention (decoder input attending to itself)
        new_x, attn = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Cross-Attention (decoder input attending to encoder output)
        new_x, cross_attn = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # Feed Forward Network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm3(x + y)

        return out, attn, cross_attn

class Decoder(nn.Module):
    """
    Stack multiple DecoderLayers.
    Final projection to prediction size is added.
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        cross_attns = []
        for layer in self.layers:
            x, attn, cross_attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)
            cross_attns.append(cross_attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, attns, cross_attns
