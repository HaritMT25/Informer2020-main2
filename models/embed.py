import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=3, m=5, pad=True):
        """
        Args:
            c_in: number of input channels
            d_model: desired output dimension per token
            tao, m: parameters for constructing the faithful vector
            pad: whether to pad the sequence before convolution
        """
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in

        # Determine number of output channels from the primary conv per input channel.
        self.kernels = int(d_model / c_in)

        # Create a convolution layer that is applied to every (m+1)-sized chunk.
        self.conv = nn.Conv1d(
            in_channels=m+1,
            out_channels=self.kernels,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        # If d_model is not exactly c_in * kernels, then we need an extra conv for the remaining channels.
        leftout_channels = d_model - c_in * self.kernels
        if leftout_channels > 0:
            self.leftout_conv = nn.Conv1d(
                in_channels=m+1,
                out_channels=leftout_channels,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            )
            nn.init.kaiming_normal_(self.leftout_conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        else:
            self.leftout_conv = None

    def construct_faithful_vec(self, t, ts_batch):
        """
        Construct a faithful vector for time t from ts_batch.
        For each channel, take values from indices t, t-tao, t-2*tao, ..., t-m*tao.
        """
        a = []
        for c in range(ts_batch.shape[1]):
            for i in range(self.m + 1):
                a.append(ts_batch[t - i * self.tao][c])
        # Return a tensor of shape [1, (m+1)*c_in]
        return torch.tensor(a, dtype=torch.float32, device=ts_batch.device).reshape(1, -1)

    def data_extract(self, ts_batch):
        """
        Extract the faithful vectors for all valid time indices from ts_batch.
        ts_batch is expected to be of shape [n_seq, c_in].
        """
        n_seq, _ = ts_batch.shape
        data_total = []
        for t in range(self.m * self.tao, n_seq):
            faithful_vec = self.construct_faithful_vec(t, ts_batch)
            data_total.append(faithful_vec)
        # data_total becomes a tensor of shape [n_new_seq, (m+1)*c_in]
        return torch.cat(data_total, dim=0)

    def forward(self, x):
        """
        Args:
            x: a numpy array or tensor of shape [batch_size, seq_len, c_in]
        
        Returns:
            x_embedded: tensor of shape [batch_size, new_seq_len, d_model]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure x is a tensor on the proper device.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        batch_size, seq_len, c_in = x.shape
        x_list = []
        # Process each batch element individually.
        for b in range(batch_size):
            ts_batch = x[b]  # Shape: [seq_len, c_in]
            extracted_data = self.data_extract(ts_batch)  # Shape: [new_seq_len, (m+1)*c_in]
            x_list.append(extracted_data)
        # Stack into tensor of shape [batch_size, new_seq_len, (m+1)*c_in]
        x_embedded = torch.stack(x_list, dim=0)

        if self.pad:
            # Pad along the time dimension (pad top with m*tao zeros).
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))

        # Split the last dimension into c_in groups of (m+1) features.
        x_split = torch.split(x_embedded, self.m + 1, dim=2)
        channel_outputs = []

        # For each channel group, apply the conv.
        for j in range(len(x_split)):
            # Permute to shape [batch_size, m+1, new_seq_len] for Conv1d.
            conv_input = x_split[j].permute(0, 2, 1)
            out = self.conv(conv_input)
            channel_outputs.append(out)
            # For the last channel, if extra output channels are needed, also apply leftout_conv.
            if j == len(x_split) - 1 and self.leftout_conv is not None:
                extra_out = self.leftout_conv(conv_input)
                channel_outputs.append(extra_out)

        # Concatenate along the channel dimension and then transpose to get [batch_size, new_seq_len, d_model].
        x_embedded_final = torch.cat(channel_outputs, dim=1).transpose(1, 2)
        return x_embedded_final



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
