import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
from .util import PositionalEncoding

class discriminator(nn.Module):
    def __init__(self, x_dim, time_step,d_model,embedding_len,device='cpu'):
        super(discriminator, self).__init__()
        self.x_dim = x_dim
        self.time_step = time_step
        self.eps = torch.finfo(torch.float).eps
        self.device = device
        self.d_model = d_model
        # Embedding and Linear Layers
        self.local_dis = nn.Linear(self.x_dim - 2 + 2 * embedding_len, self.d_model).to(device)
        
        # LSTM replaces the Transformer
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True).to(device)
        
        # Output layer
        self.dis_output = nn.Linear(self.d_model, 1).to(device)
    
    def forward(self, sample, src_mask=None):
        # Apply local_dis to the sample input
        #print(sample.shape)
        last_valid_idx = torch.sum(~src_mask, dim=1).long() - 1
        #print(last_valid_idx.shape)
        x = self.local_dis(sample)
        x = torch.relu(x)  # Non-linearity after linear layer
        #packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Apply LSTM
        xn, _ = self.lstm(x)
        batch_size = xn.size(0)
        dim_size = xn.size(2)
        #print(xn.shape)
        #last_valid_idx = last_valid_idx.view(-1, 1, 1).expand(batch_size, 1, dim)
        last_valid_idx = last_valid_idx.view(-1, 1, 1).expand(batch_size, 1, dim_size) 
        hn_last = torch.gather(xn, 1,last_valid_idx).squeeze(1)
        #hn_last = torch.gather(hn, 1, last_valid_idx).squeeze(1)
        # Sum over time step dimension (axis 1) to get a fixed-size vector
        #hn[scr_mask]
        #x = torch.relu(hn_last)
        x = self.dis_output(hn_last)
        #x = torch.sum(x, axis=1).squeeze(1)
        # Output layer to produce the final discriminator score
        #x = self.dis_output(x)
        return x









