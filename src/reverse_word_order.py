from utils import generate_reverse_word_order_data
"""
Training script for a Transformer model on a reverse word order task.
This script trains a Transformer model to reverse the order of words in sequences.
The model is trained using cross-entropy loss, which expects raw logits (unnormalized
scores) as input, not probabilities. The Transformer's output layer should produce
logits directly without applying softmax, as F.cross_entropy() applies log_softmax
internally for numerical stability.
The training process:
1. Generates synthetic data with sequences and their reversed counterparts
2. Creates a DataLoader for batch processing
3. Initializes a Transformer model
4. Trains for multiple epochs using:
    - Cross-entropy loss (expects logits from the model)
    - Adam optimizer
    - Backpropagation and gradient descent
Parameters:
     num_samples (int): Number of training samples to generate (default: 1000)
     max_seq_len (int): Maximum sequence length (default: 20)
     vocab_size (int): Size of the vocabulary (default: 50)
     embedding_dim (int): Dimension of token embeddings (default: 16)
     num_heads (int): Number of attention heads (default: 2)
     num_layers (int): Number of transformer layers (default: 2)
     batch_size (int): Batch size for training (default: 32)
Note:
     The optimizer initialization should be moved outside the training loop
     to avoid reinitializing it at every batch.
"""
from network import Transformer
from utils import WordOrderDataset

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    input_seqs, target_seqs = zip(*batch)
    
    # Pad sequences (use -1 as padding value to ignore in loss calculation)
    # Note: vocab tokens are 0 to vocab_size-1, so -1 is safe as padding
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=-1)  # Use -1 for target padding
    
    return input_padded, target_padded


if __name__ == "__main__":

    num_samples = 10
    max_seq_len = 20    
    vocab_size = 50

    embedding_dim = 16
    num_heads = 1
    num_layers = 1
    batch_size = 32

    data = generate_reverse_word_order_data(num_samples=num_samples, max_seq_len=max_seq_len, vocab_size=vocab_size)
    dataset = WordOrderDataset(data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    transformer = Transformer(vocab_size=vocab_size, embed_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers)
    
    
    # Initialize optimizer before the training loop
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    
    transformer.train()
    for epoch in range(1000):
        for batch in dataloader:
            input_seq, target_seq = batch
            output = transformer(input_seq)
            
            # Use nll_loss with ignore_index=-1 to ignore padded positions

            loss = F.nll_loss(output.view(-1, vocab_size), target_seq.view(-1), ignore_index=-1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/1000 completed")

    
    # inference example
    transformer.eval()
    test_seq = torch.tensor([[1, 2, 3, 4, 5]])  # Example input sequence
    with torch.no_grad():
        output = transformer(test_seq)
        predicted_tokens = torch.argmax(output, dim=-1)
        print("Input sequence:", test_seq)
        print("Predicted reversed sequence:", predicted_tokens)