import torch
import torch.utils.data as dataset


class WordOrderDataset(dataset.Dataset):
    def __init__(self, data):
        # Shuffle the data randomly
        indices = torch.randperm(len(data))
        self.data = [data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]