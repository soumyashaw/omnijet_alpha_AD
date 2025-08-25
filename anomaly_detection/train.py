# PyTorch imports
import torch
import torch.optim as optim
import torch.nn as nn

# Anomaly Detection class imports
from models import TokenizationModule, JetAnomalyDetector

# Example training setup
def train():
    # Create model
    tokenizer = TokenizationModule(use_pretrained_vqvae=True)
    #model = JetAnomalyDetector(verbose=True)

    """# Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy example: random input and target
    inputs = torch.randn(64, 784)  # batch of 64
    targets = torch.randint(0, 10, (64,))  # random labels

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")"""

if __name__ == "__main__":
    train()