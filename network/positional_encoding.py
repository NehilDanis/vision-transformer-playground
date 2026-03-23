# import torch    
# import torch.nn as nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, seq_size=5000):
#         super().__init__()
#         self.positional_encoding = nn.Parameter(torch.randn(seq_size, embed_dim), requires_grad=False)

#     def forward(self, x):
#         _, seq_len, _ = x.size()
#         return x + self.positional_encoding[:seq_len, :] 

import torch    
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_size=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(seq_size, embed_dim)#
    def forward(self, x):

        batch, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch, seq_len)
        return x + self.pos_embedding(positions)