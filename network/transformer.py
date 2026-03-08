import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding
from .word_embeddings import TokenEmbedding
from .attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_size=5000):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_size)
        self.layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # below the token embeddings will return a tensor of shape (batch_size, seq_len, embed_dim)
        # and the positional encoding will return a tensor of shape (seq_len, embed_dim)
        # but the plus operator will broadcast the positional encoding across the batch dimension, 
        # resulting in a tensor of shape (batch_size, seq_len, embed_dim). Meaning that each sequence 
        # in the batch will have the same positional encoding added to it.
        x = self.token_embedding(x)
        x = self.positional_encoding(x) 
        attention_percents_all_layers = []
        for layer in self.layers:
            x, attention_percents = layer(x, x, x)
            attention_percents_all_layers.append(attention_percents)

        # Map to vocab probabilities
        return F.log_softmax(self.fc_out(x), dim=-1), attention_percents_all_layers