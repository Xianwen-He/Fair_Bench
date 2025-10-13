import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np



class BinaryMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(BinaryMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # No Sigmoid, handled by BCEWithLogitsLoss
        )

    def forward(self, x):
        """
        model(x) output the logits
        """
        return self.model(x)
    
    def predict_prob(self, x):
        return torch.sigmoid(self.model(x))
    
    def predict(self, x, threshold=0.5):
        """
        label output
        """
        return (self.predict_prob(x) > 0.5).float()

