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

        # Determine if an extra convolution is needed on the last channel split.
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
        
        This forward pass mimics the functionality of the original loop-based version.
        """
        batch_size, seq_len, c_in = x.shape
        device = x.device

        # Step 1: Vectorized faithful vector extraction.
        L = seq_len - self.m * self.tao  # number of valid time steps
        valid_t = torch.arange(self.m * self.tao, seq_len, device=device)  # shape: [L]
        offset = torch.arange(0, self.m + 1, device=device) * self.tao  # shape: [m+1]
        indices = valid_t.unsqueeze(1) - offset.unsqueeze(0)  # shape: [L, m+1]
        indices = indices.unsqueeze(0).expand(batch_size, -1, -1)  # shape: [batch, L, m+1]

        # Expand indices to include the channel dimension: [batch, L, m+1, c_in]
        indices_exp = indices.unsqueeze(-1).expand(batch_size, L, self.m + 1, c_in)
        # Gather along dimension 1 directly from x (which has shape [batch, seq_len, c_in]).
        extracted = torch.gather(x, dim=1, index=indices_exp)  # shape: [batch, L, m+1, c_in]
        # Permute to [batch, L, c_in, m+1] so that the (m+1) sequence is last.
        extracted = extracted.permute(0, 1, 3, 2).contiguous()
        # Flatten the last two dimensions: shape [batch, L, c_in*(m+1)]
        x_embedded = extracted.view(batch_size, L, c_in * (self.m + 1))

        # Step 2: Optional padding on the sequence dimension.
        if self.pad:
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))

        # Step 3: Split into c_in chunks and apply convolutions.
        # Each chunk corresponds to one channel and has size (m+1)
        x_splits = torch.split(x_embedded, self.m + 1, dim=2)  # Tuple of length c_in
        conv_outputs = []
        for i, chunk in enumerate(x_splits):
            # Permute chunk to shape [batch, m+1, seq_length] for Conv1d.
            chunk = chunk.permute(0, 2, 1).contiguous()
            out_conv = self.conv(chunk)  # shape: [batch, self.kernels, seq_length]
            conv_outputs.append(out_conv)
            # For the last split, also apply leftout_conv if needed.
            if i == len(x_splits) - 1 and self.leftout_conv is not None:
                out_left = self.leftout_conv(chunk)  # shape: [batch, extra_channels, seq_length]
                conv_outputs.append(out_left)

        # Step 4: Concatenate the convolution outputs and adjust shape.
        out = torch.cat(conv_outputs, dim=1)  # shape: [batch, d_model, seq_length]
        out = out.transpose(1, 2).contiguous()  # shape: [batch, seq_length, d_model]
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

