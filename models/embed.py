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
            tao, m: parameters for constructing the “faithful vector”
            pad: whether to pad the sequence before convolution
        """
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        
        # We choose the number of output filters per channel so that 
        # c_in * kernels is (close to) d_model.
        self.kernels = int(d_model / c_in)
        
        # Create a grouped conv to apply a separate convolution on each channel’s chunk.
        # Input will be reshaped so that the number of channels is c_in*(m+1)
        # and the grouped conv will have groups=c_in so that each group is convolved separately.
        self.group_conv = nn.Conv1d(
            in_channels=c_in*(m+1),
            out_channels=c_in*self.kernels,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            groups=c_in
        )
        
        # If d_model is not an exact multiple of c_in, we need extra output channels.
        extra_channels = d_model - c_in*self.kernels
        if extra_channels > 0:
            # This conv operates on the last channel’s input (of size m+1) to produce the extra features.
            self.leftout_conv = nn.Conv1d(
                in_channels=m+1,
                out_channels=extra_channels,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            )
        else:
            self.leftout_conv = None

        # Initialize weights for group_conv and leftout_conv (if it exists)
        nn.init.kaiming_normal_(self.group_conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.leftout_conv is not None:
            nn.init.kaiming_normal_(self.leftout_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def construct_faithful_vec(self, t, ts_batch):
        """
        Construct a faithful vector at time t from the time series batch.
        The vector is constructed by taking values from t, t-tao, t-2*tao, ..., t-m*tao
        for each channel.
        
        Args:
            t: current time index (int)
            ts_batch: tensor of shape [n_seq, c_in]
        
        Returns:
            faithful_vec: tensor of shape [1, (m+1)*c_in]
        """
        a = []
        # For each channel, append m+1 time values spaced by tao.
        for c in range(ts_batch.shape[1]):
            for i in range(self.m + 1):
                a.append(ts_batch[t - i * self.tao][c])
        # Convert list to tensor of shape [1, (m+1)*c_in]
        faithful_vec = torch.tensor(a, dtype=torch.float32, device=ts_batch.device).reshape(1, -1)
        return faithful_vec

    def data_extract(self, ts_batch):
        """
        Extract faithful vectors for all valid time indices in the batch.
        
        Args:
            ts_batch: tensor of shape [n_seq, c_in]
        
        Returns:
            data_total: tensor of shape [n_seq - m*tao, (m+1)*c_in]
        """
        n_seq, _ = ts_batch.shape
        data_total = []
        for t in range(self.m * self.tao, n_seq):
            faithful_vec = self.construct_faithful_vec(t, ts_batch)
            data_total.append(faithful_vec)
        # Stack along the time dimension
        data_total = torch.cat(data_total, dim=0)
        return data_total

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: input, expected to be a numpy array or a tensor of shape [batch_size, seq_len, c_in]
        
        Returns:
            x_embedded: tensor of shape [batch_size, seq_len', d_model]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure x is a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        
        batch_size, seq_len, c_in = x.shape
        x_list = []
        # For each batch, extract the faithful vectors.
        for b in range(batch_size):
            ts_batch = x[b]
            extracted_data = self.data_extract(ts_batch)
            x_list.append(extracted_data)
        # Stack to get shape [batch_size, new_seq_len, (m+1)*c_in]
        x_embedded = torch.stack(x_list, dim=0)
        
        # Optionally pad the sequence on the top (along the time dimension)
        if self.pad:
            # Padding (left, right, top, bottom) = (0, 0, m*tao, 0)
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
        
        # At this point, each token is represented by a vector of length (m+1)*c_in.
        # Reshape so that we have c_in groups each with (m+1) features.
        # x_embedded: [batch_size, seq_len_new, c_in*(m+1)]
        # Permute to get [batch_size, c_in*(m+1), seq_len_new] for Conv1d.
        x_embedded = x_embedded.to(device)
        x_embedded = x_embedded.permute(0, 2, 1)
        
        # Apply the grouped convolution. Because groups=c_in, this is equivalent
        # to applying a separate convolution on each channel’s (m+1) features.
        conv_out = self.group_conv(x_embedded)
        # conv_out shape: [batch_size, c_in * self.kernels, seq_len_new]
        
        # If extra output channels are needed, apply the leftout convolution to the last channel's data.
        if self.leftout_conv is not None:
            # First, reshape x_embedded back to extract the last channel’s input.
            # x_embedded was [batch_size, c_in*(m+1), seq_len_new]
            # We reshape it to [batch_size, c_in, m+1, seq_len_new]
            x_split = x_embedded.view(batch_size, self.c_in, self.m+1, -1)
            # Extract the last channel’s input: shape [batch_size, m+1, seq_len_new]
            last_channel_input = x_split[:, -1, :, :]
            # Apply leftout_conv on the last channel’s input.
            extra_out = self.leftout_conv(last_channel_input)
            # Concatenate the outputs along the channel dimension.
            conv_out = torch.cat([conv_out, extra_out], dim=1)
        
        # Permute back to shape [batch_size, seq_len_new, d_model]
        x_embedded = conv_out.transpose(1, 2)
        return x_embedded


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
