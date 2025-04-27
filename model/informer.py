# model/informer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import ProbSparseAttention, ScaledDotProductAttention
from model.encoder import Encoder, EncoderLayer, ConvLayer
from model.decoder import Decoder, DecoderLayer
from utils import positional_encoding

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.norm(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = positional_encoding(5000, d_model)  # Maximum length
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.position_embedding[:seq_len, :].to(x.device)
        pos_emb = pos_emb.unsqueeze(0).repeat(x.size(0), 1, 1)  # (batch, seq_len, d_model)

        x = self.value_embedding(x) + pos_emb
        return self.dropout(x)

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=2048, dropout=0.1, attn='prob', activation='gelu', output_attention=False):
        """
        Args:
            enc_in: number of input features for encoder
            dec_in: number of input features for decoder
            c_out: output size
            seq_len: input sequence length
            label_len: label sequence length
            pred_len: prediction length
            factor: ProbSparse factor
            d_model: model dimension
            n_heads: number of attention heads
            e_layers: number of encoder layers
            d_layers: number of decoder layers
            d_ff: feedforward network dimension
            dropout: dropout rate
            attn: attention type ("prob" or "full")
            activation: activation function ("relu" or "gelu")
            output_attention: whether to output attention weights
        """
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # Attention
        Attn = ProbSparseAttention if attn == 'prob' else ScaledDotProductAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=Attn(
                        mask_flag=False,
                        factor=factor,
                        scale=None,
                        attention_dropout=dropout,
                        n_heads=n_heads,
                        d_model=d_model
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            conv_layers=[ConvLayer(d_model) for _ in range(e_layers - 1)],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=Attn(
                        mask_flag=True,
                        factor=factor,
                        scale=None,
                        attention_dropout=dropout,
                        n_heads=n_heads,
                        d_model=d_model
                    ),
                    cross_attention=Attn(
                        mask_flag=False,
                        factor=factor,
                        scale=None,
                        attention_dropout=dropout,
                        n_heads=n_heads,
                        d_model=d_model
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, enc_inp, dec_inp, enc_mask=None, dec_mask=None):
        # enc_inp: (batch, seq_len, enc_in)
        # dec_inp: (batch, label_len + pred_len, dec_in)

        enc_out = self.enc_embedding(enc_inp)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_mask)

        dec_out = self.dec_embedding(dec_inp)
        dec_out, dec_self_attns, dec_cross_attns = self.decoder(dec_out, enc_out, x_mask=dec_mask, cross_mask=enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # (batch, pred_len, c_out)
