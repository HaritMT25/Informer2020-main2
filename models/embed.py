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
        self.kernels=int(d_model/c_in)
        # Initialize the embedding layers
        self.conv = nn.Conv1d(in_channels=m+1, out_channels=self.kernels, kernel_size=3, padding=1, padding_mode='circular').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Weight initialization for the conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    # vector for each t
    def construct_faithful_vec(self, t, ts_batch, tao, m):
      # ts_batch  = [n_seq, cin]
      faithful_vec = []
      a = []
      for c in range(ts_batch.shape[1]):
        for i in range(m+1):
          a.append(ts_batch[t-i*tao][c])
      # cin*(m+1)
      return torch.tensor(a, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).reshape(1, -1)


    # vecs for all t, per batch
    def data_extract(self, ts_batch, tao=3, m=5):
      # ts_batch.shape [n_seq, cin]
      n_seq, cin = ts_batch.shape
      data_total = None
      device_used = 'cuda' if torch.cuda.is_available() else 'cpu'
      for t in range(m*tao, n_seq):
        if t == m*tao:
          data_total  = self.construct_faithful_vec(t, ts_batch, tao, m).clone().detach().to(device_used)
        else:
          new_data = self.construct_faithful_vec(t, ts_batch, tao, m).clone().detach().to(device_used)
          data_total = torch.cat((data_total, new_data), dim=0)
      # slicing
      # data_total.shape = [n_seq, cin]
      return data_total

    def forward(self, x):
        """Forward pass of the TokenEmbedding layer."""
        # expects x.type = numpy array
        batch_size, seq_len, cin = x.shape
        x_list = []

        # Ensure x is a PyTorch tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for batch_val in range(batch_size):
            ts_batch = x[batch_val]
            extracted_data = self.data_extract(ts_batch)  # Extract the faithful vectors
            x_list.append(extracted_data)


        # Convert the list back into a tensor
        x_embedded = torch.stack(x_list).to(x.device)
        if self.pad == True:
          x_embedded = F.pad(x_embedded, (0, 0, self.m*self.tao, 0))
        x_embedded1=torch.split(x_embedded,self.m+1,dim=2)
        channel_splitter=[]


        for j in range(len(x_embedded1)):
          channel_splitter.append(self.conv(x_embedded1[j].permute((0, 2, 1))))
          if j == (len(x_embedded1)-1):
            leftout_conv = nn.Conv1d(in_channels=self.m+1, out_channels=self.d_model - self.c_in*self.kernels, kernel_size=3, padding=1, padding_mode='circular').to(x.device)
            channel_splitter.append(leftout_conv(x_embedded1[j].permute((0, 2, 1))).to(x.device))



           #### concatenates d_model/c_in to avoid channel mixing
        x_embedded=torch.cat(channel_splitter,dim=1).transpose(1,2)
        #x_embedded = self.conv(x_embedded.permute((0, 2, 1))).transpose(1,2)

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

