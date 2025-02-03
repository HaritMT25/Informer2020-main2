import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=3, m=5, pad=True):
        """
        This version preserves the functionality of the second (loop‐based)
        implementation but vectorizes the extraction of the faithful vectors.
        """
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        # Compute the number of output channels per split.
        # (d_model is intended to be c_in * kernels plus possibly some extra channels.)
        self.kernels = d_model // c_in

        # Primary convolution applied to each split (operates on (m+1)-length inputs)
        self.conv = nn.Conv1d(
            in_channels=m+1, 
            out_channels=self.kernels, 
            kernel_size=3, 
            padding=1, 
            padding_mode='circular'
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        # Determine if an extra convolution is needed on the last channel split
        extra_channels = d_model - c_in * self.kernels
        if extra_channels > 0:
            self.leftout_conv = nn.Conv1d(
                in_channels=m+1, 
                out_channels=extra_channels, 
                kernel_size=3, 
                padding=1, 
                padding_mode='circular'
            )
            nn.init.kaiming_normal_(self.leftout_conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        else:
            self.leftout_conv = None

    def forward(self, x):
        """
        x: Tensor of shape [batch, seq_len, c_in]

        The forward pass mimics the second code’s behavior:
         1. For each sample and for each valid time index t (from m*tao to end),
            extract a “faithful vector” by gathering the values
            [x[t, c], x[t-tao, c], ..., x[t-m*tao, c]] for each channel c,
            with the ordering being channel‐major.
         2. Optionally pad the extracted sequence along the time dimension.
         3. Split the flattened faithful vector into c_in pieces of length (m+1)
            and apply self.conv to each. For the last channel split, if needed,
            also apply self.leftout_conv.
         4. Concatenate all convolution outputs to yield a final output with d_model channels.
        """
        batch_size, seq_len, c_in = x.shape
        device = x.device

        # ----- Step 1. Vectorized faithful vector extraction -----
        # Valid time steps start at t0 = m*tao and go up to seq_len-1.
        # Let L denote the number of valid time steps.
        L = seq_len - self.m * self.tao
        # valid_t: shape (L,)
        valid_t = torch.arange(self.m * self.tao, seq_len, device=device)
        # Offsets: we want to gather values at [t, t-tao, ..., t-m*tao].
        # offset: shape (m+1,)
        offset = torch.arange(0, self.m + 1, device=device) * self.tao
        # For each valid t, compute indices: shape (L, m+1)
        indices = valid_t.unsqueeze(1) - offset.unsqueeze(0)
        # Expand indices to apply for every sample in the batch:
        # New shape: (batch, L, m+1)
        indices = indices.unsqueeze(0).expand(batch_size, -1, -1)
        # x has shape [batch, seq_len, c_in]. We need to gather along the time dimension.
        # To use torch.gather, we expand indices to shape [batch, L, m+1, 1] then to [batch, L, m+1, c_in].
        indices_exp = indices.unsqueeze(-1).expand(batch_size, L, self.m + 1, c_in)
        # Gather: shape becomes [batch, L, m+1, c_in]
        extracted = torch.gather(x, 1, indices_exp)
        # Now, we want to reorder so that for each time step, the (m+1) values for each channel are grouped.
        # Permute to [batch, L, c_in, m+1]
        extracted = extracted.permute(0, 1, 3, 2).contiguous()
        # Flatten the last two dimensions to obtain shape [batch, L, c_in*(m+1)]
        x_embedded = extracted.view(batch_size, L, c_in * (self.m + 1))

        # ----- Step 2. Optional Padding -----
        if self.pad:
            # The original second code pads the sequence dimension at the top with (m*tao) zeros.
            # F.pad expects pad in the order (pad_left, pad_right, pad_top, pad_bottom) for 2D data.
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
            # After padding, the sequence length becomes L + m*tao == seq_len

        # ----- Step 3. Split and Convolve -----
        # The flattened faithful vector has dimension c_in*(m+1).
        # Split it into c_in chunks along the channel (last) dimension.
        # Each chunk is of size (m+1), matching the conv’s expected in_channels.
        x_splits = torch.split(x_embedded, self.m + 1, dim=2)  # Tuple of length c_in; each is [batch, seq_len, m+1]

        conv_outputs = []
        for i, chunk in enumerate(x_splits):
            # Permute chunk to shape [batch, m+1, seq_len] for Conv1d (which expects [batch, channels, length])
            chunk = chunk.permute(0, 2, 1).contiguous()
            out_conv = self.conv(chunk)  # shape: [batch, self.kernels, seq_len]
            conv_outputs.append(out_conv)
            # For the last split, also apply leftout_conv if it exists
            if i == len(x_splits) - 1 and self.leftout_conv is not None:
                out_left = self.leftout_conv(chunk)  # shape: [batch, extra_channels, seq_len]
                conv_outputs.append(out_left)

        # ----- Step 4. Concatenate and Rearrange -----
        # Concatenate along the channel dimension so that the total channels become d_model.
        out = torch.cat(conv_outputs, dim=1)  # shape: [batch, d_model, seq_len]
        # Finally, transpose to shape [batch, seq_len, d_model] (matching the second code’s output)
        out = out.transpose(1, 2).contiguous()
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

