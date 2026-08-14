import math 
import torch 
from torch.utils.data import Dataset

class SineDataset(Dataset):
    def __init__(self, num_samples=1000, noise=0):
        super().__init__()
        self.x = torch.rand(num_samples, 1)
        self.y = torch.sin(2 * math.pi * self.x) + 1 + noise * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

