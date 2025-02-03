import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=3):
        """
        c_in   : number of channels (e.g. 7)
        d_model: desired output dimension (should equal 73*c_in + 1, e.g. 512 when c_in=7)
        m      : number of previous steps to look back (here 7)
        tau    : skip (stride) between timesteps (here 3)
        """
        super(TokenEmbedding, self).__init__()
        self.m = m
        self.tau = tau
        self.c_in = c_in
        self.kernel_size = m + 1   # total number of values per channel = current + m previous = 8
        # We pad the beginning with m*tau zeros so that even early timesteps get a full window.
        self.padding = m * tau   # for m=7, tau=3, pad with 21 zeros
        self.num_kernels = 74    # total kernels
        
        # Create 74 kernels, each of size (kernel_size x 3)
        self.kernels = nn.Parameter(torch.randn(self.num_kernels, self.kernel_size, 3))
        
        # (Optional) Check that the output dimension equals d_model.
        expected_d_model = (self.num_kernels - 1) * c_in + 1
        if d_model != expected_d_model:
            raise ValueError(f"d_model should be {(self.num_kernels - 1) * c_in + 1} but got {d_model}")

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, c_in)
        Returns:
           Tensor of shape (batch_size, seq_length, d_model)
           where d_model = 73*c_in + 1 (e.g. 512 when c_in=7)
        """
        batch, seq_length, c_in = x.size()
        outputs = []  # will collect outputs of shape (batch, seq_length)

        # Process each channel with the first 73 kernels.
        for channel in range(c_in):
            # Extract data for this channel: shape (batch, seq_length)
            channel_data = x[:, :, channel]
            # Pad the beginning with zeros along the time dimension.
            # After padding, the time dimension becomes (seq_length + self.padding)
            padded = F.pad(channel_data, (self.padding, 0), mode="constant", value=0)
            # Expand padded to 4 dimensions so that its shape becomes (batch, seq_length+padding, 1, 1)
            padded = padded.unsqueeze(-1).unsqueeze(-1)
            
            # For every timestep t in the original sequence, we need to collect 8 values:
            # the value at t+padding, then t+padding - tau, t+padding - 2*tau, ... t+padding - m*tau.
            # Generate base indices for each timestep.
            t_indices = torch.arange(self.padding, self.padding + seq_length, device=x.device)  # shape (seq_length,)
            # Create offsets: 0, tau, 2*tau, ..., m*tau (length = kernel_size)
            offsets = torch.arange(0, self.kernel_size * self.tau, self.tau, device=x.device)  # shape (kernel_size,)
            # For each timestep, subtract the offsets. The resulting shape is (seq_length, kernel_size)
            window_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)
            # Expand indices to include the batch dimension: shape becomes (batch, seq_length, kernel_size)
            indices_expanded = window_indices.expand(batch, -1, -1)
            # Unsqueeze to have 4 dimensions: (batch, seq_length, kernel_size, 1)
            indices_expanded = indices_expanded.unsqueeze(-1)
            
            # Gather the required values along dim=1.
            # padded has shape (batch, seq_length+padding, 1, 1)
            # indices_expanded has shape (batch, seq_length, kernel_size, 1)
            # The result will be of shape (batch, seq_length, kernel_size, 1)
            channel_windows = padded.gather(dim=1, index=indices_expanded)
            # Squeeze out the last dimension to get shape (batch, seq_length, kernel_size)
            channel_windows = channel_windows.squeeze(-1)
            
            # Rearrange channel_windows from (batch, seq_length, kernel_size)
            # to (batch, 1, kernel_size, seq_length) for convolution.
            conv_input = channel_windows.transpose(1, 2).unsqueeze(1)
            # Now, for each of the first 73 kernels, apply a convolution.
            for k in range(self.num_kernels - 1):  # kernels 0 to 72
                # Reshape the k-th kernel to (1, 1, kernel_size, 3)
                kernel = self.kernels[k].unsqueeze(0).unsqueeze(0)
                # Convolve: input shape is (batch, 1, kernel_size, seq_length)
                # We use horizontal padding of 1 (width padding) so that the output time dimension remains seq_length.
                conv_out = F.conv2d(conv_input, kernel, stride=1, padding=(0, 1))
                # conv_out shape: (batch, 1, 1, seq_length) → squeeze to (batch, seq_length)
                conv_out = conv_out.squeeze(1).squeeze(1)
                outputs.append(conv_out)
                
        # Next, apply the 74th kernel on a randomly chosen channel.
        random_channel = torch.randint(0, c_in, (1,)).item()
        channel_data = x[:, :, random_channel]  # shape (batch, seq_length)
        padded = F.pad(channel_data, (self.padding, 0), mode="constant", value=0)
        padded = padded.unsqueeze(-1).unsqueeze(-1)  # shape: (batch, seq_length+padding, 1, 1)
        t_indices = torch.arange(self.padding, self.padding + seq_length, device=x.device)
        offsets = torch.arange(0, self.kernel_size * self.tau, self.tau, device=x.device)
        window_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)
        indices_expanded = window_indices.expand(batch, -1, -1).unsqueeze(-1)
        channel_windows = padded.gather(dim=1, index=indices_expanded)
        channel_windows = channel_windows.squeeze(-1)
        conv_input = channel_windows.transpose(1, 2).unsqueeze(1)
        kernel = self.kernels[-1].unsqueeze(0).unsqueeze(0)
        conv_out = F.conv2d(conv_input, kernel, stride=1, padding=(0, 1))
        conv_out = conv_out.squeeze(1).squeeze(1)
        outputs.append(conv_out)
        
        # Stack all outputs along the last dimension.
        # The resulting tensor will have shape (batch, seq_length, d_model)
        out_tensor = torch.stack(outputs, dim=-1)
        return out_tensor

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
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, m=7, tau=3)
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

