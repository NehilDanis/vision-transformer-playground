import torch


"""
This file contains a function to generate a dataset for the reverse word order task. 
The function generates random sequences of integers and their corresponding reversed 
sequences. Each sequence is of random length between 1 and a specified maximum length,
and the integers are randomly chosen from a specified vocabulary size.
"""
def generate_reverse_word_order_data(num_samples=1000, max_seq_len=10, vocab_size=20):
    data = []
    for _ in range(num_samples):
        seq_len = torch.randint(1, max_seq_len + 1, (1,)).item()
        sequence = torch.randint(0, vocab_size, (seq_len,))
        reversed_sequence = sequence.flip(0)
        data.append((sequence, reversed_sequence))

    return data