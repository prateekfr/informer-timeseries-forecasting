# data_loader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_len, label_len, pred_len, features, target):
        '''
        Args:
            csv_file: Path to ETT dataset (CSV file)
            seq_len: input sequence length (e.g., 96)
            label_len: label sequence length (e.g., 48)
            pred_len: prediction length (e.g., 24)
            features: list of feature column names
            target: name of the target column for prediction
        '''
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        
        # Standardize the feature columns
        self.scaler = self.standard_scaler()
        self.data_scaled = self.scaler.transform(self.data[self.features])

    def standard_scaler(self):
        """
        Fit a StandardScaler on the feature columns.
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.data[self.features])
        return scaler

    def __len__(self):
        """
        Total number of samples.
        """
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        """
        Get one sample (input sequence, output sequence).
        """
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]       # encoder input
        seq_y = self.data_scaled[r_begin:r_end]        # decoder input + prediction target

        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)

        return seq_x, seq_y
