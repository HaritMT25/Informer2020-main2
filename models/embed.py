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


        print(f"x shape: {x.shape}")        # Shape of input tensor x
        print(f"x_mark shape: {x_mark.shape}")  # Shape of time-related tensor x_mark
 
        # Value embedding
        x_val = self.value_embedding(x)
        x_pos = self.position_embedding(x)
        x_temp = self.temporal_embedding(x_mark)

        # Debug Shapes
        print(f"x_val shape (value embedding): {x_val.shape}")
        print(f"x_pos shape (positional embedding): {x_pos.shape}")
        print(f"x_temp shape (temporal embedding): {x_temp.shape}")

        # Ensure batch size consistency
        max_batch_size = max(x_val.shape[0], x_pos.shape[0], x_temp.shape[0])

        if x_val.shape[0] != max_batch_size:
            x_val = x_val.expand(max_batch_size, -1, -1)
        if x_pos.shape[0] != max_batch_size:
            x_pos = x_pos.expand(max_batch_size, -1, -1)
        if x_temp.shape[0] != max_batch_size:
            x_temp = x_temp.expand(max_batch_size, -1, -1)

        # Ensure sequence length consistency
        target_seq_len = x_val.shape[1]
        x_pos = self._adjust_sequence_length(x_pos, target_seq_len)
        x_temp = self._adjust_sequence_length(x_temp, target_seq_len)

        # Debug Shapes After Alignment
        print(f"x_val shape (after alignment): {x_val.shape}")
        print(f"x_pos shape (after alignment): {x_pos.shape}")
        print(f"x_temp shape (after alignment): {x_temp.shape}")

        # Combine embeddings
        x = x_val + x_pos + x_temp
        return self.dropout(x)

    @staticmethod
    def _adjust_sequence_length(tensor, target_seq_len):
        """
        Adjusts the sequence length of a tensor to match the target length by truncating or padding.
        """
        if tensor.shape[1] > target_seq_len:
            return tensor[:, :target_seq_len, :]  # Truncate
        elif tensor.shape[1] < target_seq_len:
            padding = torch.zeros((tensor.shape[0], target_seq_len - tensor.shape[1], tensor.shape[2]), device=tensor.device)
            return torch.cat([tensor, padding], dim=1)  # Pad
        return tensor

