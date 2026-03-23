from utils import generate_reverse_word_order_data
from network import Transformer
from utils import WordOrderDataset

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import mlflow
mlflow.set_experiment("reverse_word_order_transformer")

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention(attn_weights, head_idx=0):
    attn = attn_weights[0, head_idx].detach().cpu().numpy()  # batch 0, specified head
    
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(attn, cmap="viridis", ax=ax)

    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    ax.set_title("Attention Map")

    plt.close()

    return fig


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    input_seqs, target_seqs = zip(*batch)
    
    # Pad sequences (use -1 as padding value to ignore in loss calculation)
    # Note: vocab tokens are 0 to vocab_size-1, so -1 is safe as padding
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=-1)  # Use -1 for target padding
    
    return input_padded, target_padded


if __name__ == "__main__":

    num_samples = 1000
    max_seq_len = 5    
    vocab_size = 20

    embedding_dim = 64  # Increase from 16 to 64
    num_heads = 4       # Increase from 2 to 4
    num_layers = 2      # Increase from 1 to 2
    batch_size = 32
    num_epochs = 500   # Increase from 1000 to 2000

    lr = 0.001
    optimizer = "Adam"

    data = generate_reverse_word_order_data(num_samples=num_samples, max_seq_len=max_seq_len, vocab_size=vocab_size)
    dataset = WordOrderDataset(data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    transformer = Transformer(vocab_size=vocab_size, embed_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers)
    
    mlflow.log_params({
        "num_samples": num_samples,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_attention_heads": num_heads,
        "num_attention_layers": num_layers,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": optimizer,
        "num_epochs": num_epochs
    })
    
    # Initialize optimizer before the training loop
    if optimizer == "Adam":
        optimizer = optim.Adam(transformer.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = optim.SGD(transformer.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    transformer.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_seq, target_seq = batch
            output, attention_percents_all_layers = transformer(input_seq)
            if (epoch + 1) % 100 == 0:
                for layer_idx, attention_percents in enumerate(attention_percents_all_layers):
                    for head_idx in range(attention_percents.shape[1]):
                        fig = plot_attention( attention_percents, head_idx=head_idx)  # Plot each head for this layer
                        mlflow.log_figure(fig, f"attention_layer_{layer_idx}_head_{head_idx}_epoch_{epoch + 1}.png")
            loss = F.nll_loss(output.view(-1, vocab_size), target_seq.view(-1), ignore_index=-1)
            # print(f"Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step=epoch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs} completed")

    # save the model and load later to visualize attention on test examples
    model_path = "reverse_word_order_transformer.pth"
    torch.save(transformer.state_dict(), model_path)
    mlflow.log_artifact(model_path)
    # inference example

    single_input = data[0][0].unsqueeze(0)  # Get the first input sequence and add batch dimension
    transformer.eval()
    test_seq = torch.tensor(single_input)  # Example input sequence
    with torch.no_grad():
        output, attention_percents_all_layers = transformer(test_seq)
        for layer_idx, attention_percents in enumerate(attention_percents_all_layers):
                for head_idx in range(attention_percents.shape[1]):
                    fig = plot_attention(attention_percents, head_idx=head_idx)  # Plot each head for this layer
                    mlflow.log_figure(fig, f"inference_attention_layer_{layer_idx}_head_{head_idx}_epoch_{epoch + 1}.png")
        predicted_tokens = torch.argmax(output, dim=-1)
        print("Input sequence:", test_seq)
        print("Predicted reversed sequence:", predicted_tokens)