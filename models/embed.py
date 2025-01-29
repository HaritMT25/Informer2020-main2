import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=3):
        super(TokenEmbedding, self).__init__()
        self.m = m          # Number of past timesteps to consider
        self.tau = tau      # Stride between timesteps
        self.c_in = c_in    # Number of input channels
        self.kernel_size = m + 1  # Current + m past values
        
        # Generate 74 kernels of shape (kernel_size, 3)
        self.num_kernels = 74
        self.kernels = nn.Parameter(torch.randn(self.num_kernels, self.kernel_size, 3))
        
        # Calculate padding needed at the beginning of the sequence
        self.pad_size = m * tau
        self.pad = nn.ConstantPad1d((self.pad_size, 0), 0)

    def forward(self, x):
        # Input x: (batch_size, seq_len, c_in)
        batch_size, seq_len, _ = x.size()
        
        # Pad the beginning of the sequence
        x_padded = self.pad(x.permute(0, 2, 1))  # (batch_size, c_in, seq_len + pad_size)
        x_padded = x_padded.permute(0, 2, 1)     # (batch_size, seq_len + pad_size, c_in)
        
        # Create sliding windows with tau stride
        windows = x_padded.unfold(1, self.kernel_size, self.tau)  # (batch_size, num_windows, c_in, kernel_size)
        windows = windows.permute(0, 2, 1, 3)  # (batch_size, c_in, num_windows, kernel_size)
        
        # Prepare for convolutions (add channel dimension)
        windows = windows.unsqueeze(2)  # (batch_size, c_in, 1, num_windows, kernel_size)
        
        # Apply first 73 kernels to all channels
        conv_outputs = []
        for k in range(self.num_kernels - 1):
            kernel = self.kernels[k].view(1, 1, 1, self.kernel_size, 3)  # (1,1,1,8,3)
            kernel = kernel.repeat(batch_size, self.c_in, 1, 1, 1)       # (batch, c_in, 1, 8, 3)
            
            # Perform depthwise convolution
            output = F.conv3d(windows, kernel, groups=self.c_in*batch_size)
            conv_outputs.append(output.squeeze(-3))
        
        # Apply 74th kernel to random channel
        rand_ch = random.randint(0, self.c_in-1)
        selected_ch = windows[:, rand_ch:rand_ch+1]  # (batch_size, 1, 1, num_windows, kernel_size)
        kernel = self.kernels[-1].view(1, 1, 1, self.kernel_size, 3)
        output = F.conv3d(selected_ch, kernel.repeat(batch_size, 1, 1, 1, 1), groups=batch_size)
        conv_outputs.append(output.squeeze(-3).squeeze(1))
        
        # Combine outputs
        combined = torch.cat([o.view(batch_size, -1) for o in conv_outputs], dim=1)
        
        # Match sequence length and reshape to d_model
        final = combined[:, :seq_len*self.num_kernels].view(batch_size, seq_len, -1)
        
        # Project to d_model if needed
        if final.size(-1) != d_model:
            final = nn.Linear(final.size(-1), d_model)(final)
            
        return final

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Rest of your embedding classes (Temporal, DataEmbedding, etc.) remain similar


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
