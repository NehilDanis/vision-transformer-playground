from unittest import result

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # torch.manual_seed(42)
        self.query_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.key_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.value_weights = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, encoding_q, encoding_k, encoding_v, mask=None):
        Q = self.query_weights(encoding_q)
        K = self.key_weights(encoding_k)
        V = self.value_weights(encoding_v)

        # here we give numbers in the transpose because the data might consists batches and multiple heads, so we want to transpose only the last two dimensions
        similarities = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1)))

        if mask is not None:
            # below function takes the attention scores and replaces the elements where the mask is True with -1e9.
            #  This is done to effectively ignore those positions when applying the softmax function later, as the 
            # large negative value will result in a near-zero probability for those positions.
            similarities = similarities.masked_fill(mask, -1e9)

        # applying the softmax to the scaled similarities, determines the percentage of attention that each token should pay to the other tokens in the sequence
        attention_percents = F.softmax(similarities, dim=-1)
        attention_scores = torch.matmul(attention_percents, V)
        return attention_scores, attention_percents
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()

        # create bunch of attention heads
        self.heads = nn.ModuleList([Attention(embed_dim) for _ in range(num_heads)])

        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, encoding_q, encoding_k, encoding_v):
        ## run the data through all of the attention heads

        concatanated_attention_scores = torch.cat([head(encoding_q, encoding_k, encoding_v)[0] for head in self.heads], dim=-1)
        
        # calculate the attention percentages for each head and stack them together as well, to visualize them later
        # shape batch_size, num_heads, seq_len, seq_len -> stacked_attention_percents
        # shape batch_size, seq_len, embed_dim -> self.proj(concatanated_attention_scores)
        stacked_attention_percents = torch.stack([head(encoding_q, encoding_k, encoding_v)[1] for head in self.heads], dim=1)
        return self.proj(concatanated_attention_scores), stacked_attention_percents