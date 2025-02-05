import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=12, m=7, pad=True):
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        self.kernels = int(d_model / c_in)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the embedding layers on the proper device
        self.conv = nn.Conv1d(in_channels=m+1, out_channels=self.kernels, 
                              kernel_size=3, padding=1, padding_mode='circular').to(self.device)
        self.leftout_conv = nn.Conv1d(in_channels=m+1, 
                                      out_channels=self.d_model - self.c_in * self.kernels, 
                                      kernel_size=3, padding=1, padding_mode='circular').to(self.device)

        # Weight initialization for all conv layers
        for m_module in self.modules():
            if isinstance(m_module, nn.Conv1d):
                nn.init.kaiming_normal_(m_module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def data_extract(self, ts_batch, tao=12, m=7):
        """
        Vectorized extraction of faithful vectors.
        
        Given a time-series batch tensor `ts_batch` of shape [n_seq, c_in],
        this method returns a tensor of shape [n_seq - m*tao, c_in*(m+1)].
        
        For each valid time index t (from m*tao to n_seq-1), for each channel c,
        the extracted vector is:
            [ts_batch[t][c], ts_batch[t-tao][c], ..., ts_batch[t-m*tao][c]]
        and these values for all channels are concatenated in channel-major order.
        """
        # ts_batch is assumed to be already on self.device with shape (n_seq, c_in)
        n_seq, cin = ts_batch.shape
        n_valid = n_seq - m * tao  # valid time indices: t = m*tao, m*tao+1, ..., n_seq-1

        # Create a vector of valid t indices on device: shape (n_valid,)
        #keep in mind that changing here m*tao to m-1*tao
        t_indices = torch.arange((m-1) * tao, n_seq, device=ts_batch.device)  # t values

        # For each valid t, create the offsets: t, t-tao, ..., t-m*tao.
        # offsets: shape (m+1,)
        offsets = torch.arange(0, m + 1, device=ts_batch.device) * tao  
        # time_indices: shape (n_valid, m+1) where each row is: [t, t-tao, ..., t-m*tao]
        time_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)

        # For each channel, gather the corresponding values.
        # Create a channel index tensor: shape (1, cin, 1) expanded to (n_valid, cin, m+1)
        channel_idx = torch.arange(cin, device=ts_batch.device).view(1, cin, 1).expand(n_valid, cin, m + 1)
        # Expand time_indices to shape (n_valid, 1, m+1)
        time_idx_expanded = time_indices.unsqueeze(1).expand(n_valid, cin, m + 1)
        # Using advanced indexing on ts_batch (shape: [n_seq, c_in]) to get a tensor of shape (n_valid, cin, m+1)
        extracted = ts_batch[time_idx_expanded, channel_idx]
        # Flatten the last two dimensions in channel-major order:
        # For each t, the order is: for c=0: [ts[t][0], ts[t-tao][0], ..., ts[t-m*tao][0],
        #                                then for c=1: [ts[t][1], ts[t-tao][1], ..., ts[t-m*tao][1], ...]
        faithful_vec = extracted.reshape(n_valid, cin * (m + 1))
        return faithful_vec

    def forward(self, x):
        """Forward pass of the TokenEmbedding layer.
        
        Input x is expected to be a numpy array or a tensor of shape [batch_size, seq_len, c_in].
        The output is computed such that the numerical results (including the ordering)
        are identical to the original code.
        """
        batch_size, seq_len, cin = x.shape
        x_list = []

        # Convert input x to a tensor on the proper device if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        # Process each batch entry separately
        for batch_val in range(batch_size):
            ts_batch = x[batch_val]  # shape: (seq_len, c_in)
            extracted_data = self.data_extract(ts_batch, self.tao, self.m)  # shape: (n_seq - m*tao, c_in*(m+1))
            x_list.append(extracted_data)

        # Stack along the batch dimension: shape (batch_size, n_valid, c_in*(m+1))
        x_embedded = torch.stack(x_list)

        # Padding along the time dimension if required
        if self.pad:
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
        # Split the last dimension into chunks of size (m+1) to separate channels
        x_embedded1 = torch.split(x_embedded, self.m + 1, dim=2)
        channel_splitter = []

        # Process each channel-split with the corresponding convolution
        for j in range(len(x_embedded1)):
            # Permute to shape (batch_size, m+1, time) as expected by nn.Conv1d
            conv_in = x_embedded1[j].permute(0, 2, 1)
            channel_splitter.append(self.conv(conv_in))
            if j == (len(x_embedded1) - 1):
                channel_splitter.append(self.leftout_conv(conv_in))

        # Concatenate along the channel dimension and transpose back to time-major order
        x_embedded = torch.cat(channel_splitter, dim=1).transpose(1, 2)
        return x_embedded



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x)

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

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
        return month_x + day_x + weekday_x + hour_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
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
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, m=7, tao=12)
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
