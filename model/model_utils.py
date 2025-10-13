import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def train_BinaryMLP(model, loader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch, w_batch in loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            y_pred = model(X_batch)

            # Compute loss (per-sample) and apply custom weights
            loss = criterion(y_pred, y_batch)
            weighted_loss = (loss * w_batch).mean()  # Apply weights and average

            # Backward pass and optimize
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(loader):.4f}")


def collect_predictions(model, loader, device):
    model.eval()
    all_probs = []  # Store predictions
    all_labels = []       # Store true labels (optional)

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.to(device)

            # Get model output (logits) and convert to probabilities
            # logits = model(X_batch)
            probabilities =  model.predict_prob(X_batch)
            labels = model.predict(X_batch)

            # Store predictions and labels
            all_probs.extend(probabilities.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def evaluate(model, loader, device):
    """
    naive function to get accuracy with threshold as 0.5
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Predict and apply sigmoid for probabilities
            y_pred = torch.sigmoid(model(X_batch))
            y_pred = (y_pred > 0.5).float()  # Convert to binary predictions

            correct += (y_pred == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Accuracy: {correct / total:.4f}")



def prepare_dataloader(X, y, weights=None, batch_size=32, shuffle=False):
    """
    Input: 
        X, y weights: numpy array
    Output:
        dataset, dataloader
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    if weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float32).flatten()
    else:
        weights = np.ones(X.shape[0])
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    data_set = TensorDataset(X_tensor, y_tensor, weights_tensor)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    return data_set, data_loader