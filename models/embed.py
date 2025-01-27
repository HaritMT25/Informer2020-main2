import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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


class FeatureExtractor(nn.Module):
    def __init__(self, c_in, m=7, tau=3):
        super(FeatureExtractor, self).__init__()
        self.m = m
        self.tau = tau
        self.window_size = m + 1  # Current time step + m * tau
        self.c_in = c_in

    def forward(self, x):
        batch_size, seq_length, c_in = x.shape
        assert c_in == self.c_in, "Input channels do not match initialization."

        valid_t = seq_length - (self.m * self.tau)
        feature_vectors = []

        for t in range(valid_t):
            indices = [t + i * self.tau for i in range(self.m + 1)]
            window = x[:, indices, :]  # Shape: (batch_size, m+1, c_in)
            flat_window = window.reshape(batch_size, -1)  # Shape: (batch_size, c_in * (m+1))
            feature_vectors.append(flat_window)

        output = torch.stack(feature_vectors, dim=1)  # Shape: (batch_size, valid_t, c_in * (m+1))
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=3):
        super(TokenEmbedding, self).__init__()
        self.feature_extractor = FeatureExtractor(c_in, m=m, tau=tau)
        self.linear = nn.Linear(c_in * (m + 1), d_model)

    def forward(self, x):
        features = self.feature_extractor(x)  # Shape: (batch_size, valid_t, c_in * (m+1))
        embeddings = self.linear(features)  # Shape: (batch_size, valid_t, d_model)
        return embeddings


class CircularConv1D(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(CircularConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def circular_pad(self, x):
        padding_size = self.kernel_size - 1
        # Add padding to the start and end
        x = torch.cat([x[:, :, -padding_size:], x, x[:, :, :padding_size]], dim=2)
        return x

    def forward(self, x):
        original_seq_len = x.size(1)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, c_in, seq_len)
        x = self.circular_pad(x)
        x = self.conv(x)
        x = x[:, :, :original_seq_len]  # Truncate to match the original sequence length
        return x.permute(0, 2, 1)  # Shape: (batch_size, seq_len, d_model)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(c_in, d_model)  # Placeholder for TemporalEmbedding
        self.circular_conv = CircularConv1D(c_in=d_model, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Debugging shapes
        print(f"Input x shape: {x.shape}")
        print(f"Input x_mark shape: {x_mark.shape}")

        # Apply token and temporal embeddings
        x_value = self.value_embedding(x)  # Shape: (batch_size, seq_len, d_model)
        print(f"x_value shape after TokenEmbedding: {x_value.shape}")

        x_temporal = self.temporal_embedding(x_mark)  # Shape: (batch_size, seq_len, d_model)
        print(f"x_temporal shape after TemporalEmbedding: {x_temporal.shape}")

        x = x_value + x_temporal

        # Apply CircularConv1D
        x = self.circular_conv(x)  # Shape: (batch_size, seq_len, d_model)
        print(f"x shape after CircularConv1D: {x.shape}")

        # Add positional embedding
        x_pos = self.position_embedding(x)  # Shape: (batch_size, seq_len, d_model)
        print(f"x_pos shape: {x_pos.shape}")

        x = x + x_pos

        # Apply dropout
        x = self.dropout(x)

        print(f"Final output shape: {x.shape}")
        return x
