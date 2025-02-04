import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=3, m=5, pad=True):
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        self.kernels = d_model // c_in

        # Initialize the embedding layers
        self.conv = nn.Conv1d(
            in_channels=m+1,
            out_channels=self.kernels,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )

        self.leftout_conv = nn.Conv1d(
            in_channels=m+1,
            out_channels=d_model - c_in * self.kernels,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )

        # Weight initialization for the conv layers
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def construct_faithful_vec(self, t, ts_batch):
        # Efficiently construct the faithful vector using tensor operations
        indices = torch.arange(t, t - (self.m + 1) * self.tao, -self.tao, device=ts_batch.device)
        return ts_batch[indices].reshape(1, -1)

    def data_extract(self, ts_batch):
        n_seq, _ = ts_batch.shape
        data_total = torch.stack(
            [self.construct_faithful_vec(t, ts_batch) for t in range(self.m * self.tao, n_seq)]
        )
        return data_total.squeeze(1)

    def forward(self, x):
        batch_size, seq_len, cin = x.shape
        device = x.device

        x_list = [self.data_extract(x[batch_val]) for batch_val in range(batch_size)]
        x_embedded = torch.stack(x_list).to(device)

        if self.pad:
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))

        x_embedded1 = torch.split(x_embedded, self.m + 1, dim=2)
        channel_splitter = [
            self.conv(chunk.permute(0, 2, 1)) for chunk in x_embedded1
        ]

        if self.d_model % self.c_in != 0:
            channel_splitter.append(
                self.leftout_conv(x_embedded1[-1].permute(0, 2, 1))
            )

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
