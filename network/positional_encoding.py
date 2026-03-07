import torch    
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_size=5000):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(seq_size, embed_dim), requires_grad=False)

    def forward(self, x):
        _, seq_len, _ = x.size()
        return x + self.positional_encoding[:seq_len, :] 