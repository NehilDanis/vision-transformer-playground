import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        torch.manual_seed(42)
        self.query_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.key_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.value_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, token_encodings):
        Q = self.query_weights(token_encodings)
        K = self.key_weights(token_encodings)
        V = self.value_weights(token_encodings)

        # here we give numbers in the transpose because the data might consists batches and multiple heads, so we want to transpose only the last two dimensions
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1)))

        # applying the softmax to the scaled similarities, determines the percentage of attention that each token should pay to the other tokens in the sequence
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output
