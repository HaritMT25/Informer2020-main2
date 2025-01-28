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

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, c_in=7, n=17420, d_model=512, num_kernels=74, kernel_size=(8, 3)):
        super(TokenEmbedding, self).__init__()
        self.c_in = c_in
        self.n = n
        self.d_model = d_model
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size

        # Initialize 74 random kernels of size 8x3
        self.kernels = nn.Parameter(torch.randn(num_kernels, *kernel_size))

    def forward(self, x):
        # Input x: (batch_size, seq_length, c_in, 8)
        batch_size, seq_length, c_in, _ = x.shape

        # Flatten the input data for each timestep
        x_flat = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, c_in * 8)

        # Initialize a list to store the results of each kernel application
        outputs = []

        # Apply the first 73 kernels to each channel
        for channel in range(c_in):
            for kernel_idx in range(self.num_kernels - 1):  # Use first 73 kernels
                kernel = self.kernels[kernel_idx].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8, 3)
                
                # Extract the required values for each timestep
                channel_data = x[:, :, channel, :].unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, 1, seq_length, 8, 1)
                
                # Create the desired input vector for each timestep
                input_vectors = []
                for t in range(seq_length):
                    # Get the current and future values with a skip of 1
                    indices = [t + i for i in range(self.kernel_size[1]) if t + i < seq_length]
                    if len(indices) < self.kernel_size[1]:
                        # Pad with zeros if there are not enough future values
                        pad_size = self.kernel_size[1] - len(indices)
                        indices += [seq_length - 1] * pad_size
                    input_vector = channel_data[:, :, indices, :]  # Shape: (batch_size, 1, kernel_size[1], 8, 1)
                    input_vectors.append(input_vector)
                
                # Stack the input vectors along the sequence length dimension
                input_vectors = torch.cat(input_vectors, dim=2)  # Shape: (batch_size, 1, seq_length, kernel_size[1], 8, 1)
                
                # Apply 2D convolution with stride=1 and padding to maintain sequence length
                conv_output = F.conv2d(
                    input_vectors.squeeze(-1).squeeze(1),  # Shape: (batch_size, seq_length, kernel_size[1], 8)
                    kernel,  # Shape: (1, 1, 8, 3)
                    stride=1,
                    padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # Padding to maintain spatial dimensions
                )  # Shape: (batch_size, seq_length, 1, 1)
                outputs.append(conv_output.squeeze(-1).squeeze(-1))  # Shape: (batch_size, seq_length)

        # Apply the 74th kernel to a random channel
        random_channel = torch.randint(0, c_in, (1,)).item()
        kernel = self.kernels[-1].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8, 3)
        channel_data = x[:, :, random_channel, :].unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, 1, seq_length, 8, 1)
        
        # Create the desired input vector for each timestep
        input_vectors = []
        for t in range(seq_length):
            # Get the current and future values with a skip of 1
            indices = [t + i for i in range(self.kernel_size[1]) if t + i < seq_length]
            if len(indices) < self.kernel_size[1]:
                # Pad with zeros if there are not enough future values
                pad_size = self.kernel_size[1] - len(indices)
                indices += [seq_length - 1] * pad_size
            input_vector = channel_data[:, :, indices, :]  # Shape: (batch_size, 1, kernel_size[1], 8, 1)
            input_vectors.append(input_vector)
        
        # Stack the input vectors along the sequence length dimension
        input_vectors = torch.cat(input_vectors, dim=2)  # Shape: (batch_size, 1, seq_length, kernel_size[1], 8, 1)
        
        # Apply 2D convolution with stride=1 and padding to maintain sequence length
        conv_output = F.conv2d(
            input_vectors.squeeze(-1).squeeze(1),  # Shape: (batch_size, seq_length, kernel_size[1], 8)
            kernel,  # Shape: (1, 1, 8, 3)
            stride=1,
            padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # Padding to maintain spatial dimensions
        )  # Shape: (batch_size, seq_length, 1, 1)
        outputs.append(conv_output.squeeze(-1).squeeze(-1))  # Shape: (batch_size, seq_length)

        # Concatenate all outputs along the last dimension
        output = torch.stack(outputs, dim=-1)  # Shape: (batch_size, seq_length, 74 * c_in + 1)

        # Reshape to (batch_size, seq_length, d_model)
        output = output.reshape(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, d_model)

        return output


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
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

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
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
