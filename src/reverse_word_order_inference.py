import torch
import torch.nn as nn
import torch.nn.functional as F

from network import Transformer




if __name__ == "__main__":
    """
    This script takes sequence of numbers and returns them in reverse order using inference.
    It loads a trained transformer model and visualizes the attention maps for a test input sequence.
    """

    model_path = "reverse_word_order_transformer.pth"
    transformer = Transformer(vocab_size=20, embed_dim=64, num_heads=4, num_layers=2)
    transformer.load_state_dict(torch.load(model_path))
    transformer.eval()

    # Example input sequence
    test_seq = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        output, attention_percents_all_layers = transformer(test_seq)
        predicted_tokens = torch.argmax(output, dim=-1)
        print("Input sequence:", test_seq)
        print("Predicted reversed sequence:", predicted_tokens)