import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

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

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # Clamp the values to be within the valid range for each embedding layer
        minute_x = self.minute_embed(torch.clamp(x[:,:,3], 0, 3)) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(torch.clamp(x[:,:,2], 0, 23))
        weekday_x = self.weekday_embed(torch.clamp(x[:,:,1], 0, 6))
        day_x = self.day_embed(torch.clamp(x[:,:,0], 0, 31))

        return hour_x + weekday_x + day_x  + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Ensure input tensor x is float32
        x = x.type(torch.float32)

        # Adjust x_mark's sequence length to match x
        x_mark = x_mark[:, :x.shape[1], :]

        # Apply embeddings
        x_val = self.value_embedding(x)  # Value embedding
        x_pos = self.position_embedding(x)  # Positional embedding
        x_temp = self.temporal_embedding(x_mark)  # Temporal embedding

        # Ensure sequence lengths match
        if x_pos.shape[1] > x_val.shape[1]:
            x_pos = x_pos[:, :x_val.shape[1], :]  # Truncate positional embedding
        elif x_pos.shape[1] < x_val.shape[1]:
            padding = torch.zeros((x_pos.shape[0], x_val.shape[1] - x_pos.shape[1], x_pos.shape[2]), device=x_pos.device)
            x_pos = torch.cat([x_pos, padding], dim=1)  # Pad positional embedding

        if x_temp.shape[1] > x_val.shape[1]:
            x_temp = x_temp[:, :x_val.shape[1], :]  # Truncate temporal embedding
        elif x_temp.shape[1] < x_val.shape[1]:
            padding = torch.zeros((x_temp.shape[0], x_val.shape[1] - x_temp.shape[1], x_temp.shape[2]), device=x_temp.device)
            x_temp = torch.cat([x_temp, padding], dim=1)  # Pad temporal embedding

        # Debugging shapes (optional, remove in production)
        print(f"x_val shape: {x_val.shape}")
        print(f"x_pos shape: {x_pos.shape}")
        print(f"x_temp shape: {x_temp.shape}")

        # Combine the embeddings
        x = x_val + x_pos + x_temp

        return self.dropout(x)
