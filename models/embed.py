import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=3, m=5, pad=True):
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        self.kernels = d_model // c_in
        self.remainder = d_model % c_in
        
        self.valid_seq_len = lambda seq_len: seq_len - m*tao
        
        self.conv = nn.Conv1d(m+1, self.kernels, kernel_size=3, 
                             padding=1, padding_mode='circular')
        
        if self.remainder > 0:
            self.conv_remainder = nn.Conv1d(m+1, 1, kernel_size=3, 
                                           padding=1, padding_mode='circular')
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def create_sliding_windows(self, x):
        batch_size, seq_len, _ = x.shape
        window_size = self.m + 1
        
        indices = torch.arange(self.m*self.tao, seq_len, device=x.device)
        window_indices = indices.unsqueeze(1) - torch.arange(0, window_size*self.tao, self.tao, device=x.device).flip(0)
        
        x = x.gather(1, window_indices.view(1, -1).expand(batch_size, -1).unsqueeze(-1).expand(-1, -1, self.c_in))
        return x.view(batch_size, -1, window_size, self.c_in).permute(0, 3, 2, 1).reshape(batch_size*self.c_in, window_size, -1)

    def forward(self, x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=self.conv.weight.device)
    
    batch_size, seq_len, _ = x.shape
    x = x.to(self.conv.weight.device)
    
    # Create sliding windows: shape [batch*c_in, m+1, valid_seq]
    x_windows = self.create_sliding_windows(x)
    
    # Apply the main convolution: output shape [batch*c_in, kernels, valid_seq]
    conv_out = self.conv(x_windows)
    # Reshape to [batch, c_in, kernels, valid_seq]
    conv_out_reshaped = conv_out.view(batch_size, self.c_in, self.kernels, -1)
    
    if self.remainder > 0:
        # Process the remainder channels using conv_remainder.
        # Select the first `remainder` channels.
        x_windows_remainder = x_windows.view(batch_size, self.c_in, self.m + 1, -1)[:, :self.remainder, :, :]
        # Merge batch and remainder dimensions.
        x_windows_remainder = x_windows_remainder.reshape(-1, self.m + 1, x_windows_remainder.shape[-1])
        # Apply the remainder convolution: shape [batch*remainder, 1, valid_seq]
        rem_out = self.conv_remainder(x_windows_remainder)
        # Reshape to [batch, remainder, 1, valid_seq]
        rem_out_reshaped = rem_out.view(batch_size, self.remainder, 1, -1)
        
        # For the first `remainder` channels, concatenate along the "kernel" dimension:
        #   conv_out_reshaped[:, :remainder] has shape [batch, remainder, kernels, valid_seq]
        #   rem_out_reshaped has shape [batch, remainder, 1, valid_seq]
        first_embed = torch.cat([conv_out_reshaped[:, :self.remainder], rem_out_reshaped], dim=2)
        # For the remaining channels, keep the original conv outputs.
        if self.c_in - self.remainder > 0:
            second_embed = conv_out_reshaped[:, self.remainder:]  # shape: [batch, c_in - remainder, kernels, valid_seq]
            # Flatten channel and kernel dimensions separately for both parts.
            first_embed = first_embed.reshape(batch_size, -1, first_embed.shape[-1])
            second_embed = second_embed.reshape(batch_size, -1, second_embed.shape[-1])
            # Concatenate along the feature dimension.
            out = torch.cat([first_embed, second_embed], dim=1)  # shape: [batch, d_model, valid_seq]
        else:
            # Only remainder channels exist.
            out = first_embed.reshape(batch_size, -1, first_embed.shape[-1])
    else:
        # If no remainder, just reshape.
        out = conv_out_reshaped.reshape(batch_size, -1, conv_out_reshaped.shape[-1])
    
    # Permute to get shape [batch, valid_seq, d_model]
    out = out.permute(0, 2, 1)
    
    if self.pad:
        # Pad along the sequence length dimension at the beginning if needed.
        out = F.pad(out, (0, 0, self.m*self.tao, 0))
    
    return out

#############################################
# 2. (Optional) Other Embedding Modules
#############################################
# Here we include two additional embeddings (positional and temporal) so that we can combine them in a DataEmbedding.
# These follow the common transformer practice.

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it is saved with the model but not updated by gradients.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is assumed to have shape (batch, seq_length, *)
        return self.pe[:, :x.size(1)]

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        # Create a sinusoidal embedding table for c_in different categories.
        w = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # x should be indices with shape (batch, seq_length, *).
        return self.emb(x)

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        # For simplicity, we assume fixed sizes for temporal features.
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        # Use FixedEmbedding if embed_type=='fixed'
        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        # If frequency 't' (e.g. minute-level), include minute embedding.
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        # Assume x has shape (batch, seq_length, 5) corresponding to (month, day, weekday, hour, minute)
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return month_x + day_x + weekday_x + hour_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

#############################################
# 3. DataEmbedding: combining value, positional, and temporal embeddings.
#############################################

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        c_in: number of channels (e.g., 7)
        d_model: desired embedding dimension (e.g., 512, which must match the TokenEmbedding output)
        embed_type: either 'fixed' (using sinusoidal temporal embeddings) or 'timeF'
        freq: frequency indicator used for temporal embedding (e.g., 'h')
        dropout: dropout rate.
        """
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, m=5, tao=3)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x: tensor of shape (batch, seq_length, c_in) – the raw input channels.
        x_mark: tensor of shape (batch, seq_length, 5) – temporal feature markers,
                for example, (month, day, weekday, hour, minute).
        """
        # Sum the three embeddings.
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

