import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=1):  # Modified tau=1 to maintain sequence length
        super(TokenEmbedding, self).__init__()
        self.m = m
        self.tau = tau
        self.c_in = c_in
        self.kernel_size = m + 1
        self.num_kernels = 74
        self.pad_size = m * tau
        self.pad = nn.ConstantPad1d((self.pad_size, 0), 0)
        self.kernels = nn.Parameter(torch.randn(self.num_kernels, self.kernel_size))
        self.proj = nn.Conv1d(73 * c_in + 1, d_model, kernel_size=1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_padded = self.pad(x.permute(0, 2, 1))  # [batch, c_in, seq + pad]
        
        # Create sliding windows with stride=tau=1
        windows = x_padded.unfold(
            dimension=2,
            size=self.kernel_size,
            step=self.tau
        )  # [batch, c_in, num_windows, kernel_size]
        num_windows = windows.shape[2]
        
        # Process first 73 kernels
        windows = windows.permute(0, 2, 1, 3).reshape(-1, self.c_in, self.kernel_size)
        main_outputs = []
        for k in range(self.num_kernels-1):
            kernel = self.kernels[k].repeat(self.c_in, 1, 1)  # [c_in, 1, kernel]
            out = F.conv1d(windows, kernel, groups=self.c_in)
            main_outputs.append(out.view(batch_size, num_windows, self.c_in))
        
        # Process 74th kernel
        rand_ch = torch.randint(0, self.c_in, (1,)).item()
        selected = windows[:, rand_ch:rand_ch+1, :]
        final_feature = F.conv1d(selected, self.kernels[-1][None, None, :])
        final_feature = final_feature.view(batch_size, num_windows, 1)
        
        # Combine and project
        combined = torch.cat([torch.cat(main_outputs, dim=2), final_feature], dim=2)
        return self.proj(combined.permute(0, 2, 1))  # [batch, d_model, seq_len]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]  # [1, seq_len, d_model]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24; weekday_size = 7; day_size = 32; month_size = 13
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
        return (self.hour_embed(x[:, :, 3]) + 
                self.weekday_embed(x[:, :, 2]) + 
                self.day_embed(x[:, :, 1]) + 
                self.month_embed(x[:, :, 0]) + 
                minute_x).permute(0, 2, 1)  # [batch, d_model, seq_len]

class DataEmbedding(nn.Module):
    def _init_(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self)._init_()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        value_emb = self.value_embedding(x)
        
        pos_emb = self.position_embedding(x)
        
        # Check if temporal embedding and value embedding have the same size in dim 2
        # if not, we pad temporal embedding with zeros to match the size of value embedding
        
        temp_emb = self.temporal_embedding(x_mark)

        if temp_emb.shape[1] != value_emb.shape[1]:
            # Pad temporal embedding with zeros to match the size of value embedding
            pad_size = value_emb.shape[1] - temp_emb.shape[1]
            temp_emb = torch.cat([temp_emb, torch.zeros(temp_emb.shape[0], pad_size, temp_emb.shape[2]).to(temp_emb.device)], dim=1)

        return self.dropout(value_emb + pos_emb + temp_emb)
