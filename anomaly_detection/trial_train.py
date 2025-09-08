import torch
import awkward as ak
from torch.utils.data import DataLoader
from models import JetAnomalyDetector, JetDataset

# Example: Prepare your tokenized data (replace with your actual data loading/tokenization)
# Assume tokenized_data is a dict: {"jet1": torch.LongTensor, "jet2": torch.LongTensor}
# and labels is a torch.Tensor of shape (N, 1)
tokenized_data = {
    "jet1": torch.randint(0, 8194, (100, 128)),  # (batch, seq_len)
    "jet2": torch.randint(0, 8194, (100, 128))
}
awkward_data = {
    "jet1": ak.Array(tokenized_data["jet1"].cpu().numpy()),
    "jet2": ak.Array(tokenized_data["jet2"].cpu().numpy())
}

print("Dimensionality:", awkward_data["jet1"][1].ndim)  # Should print 2

labels = torch.randint(0, 2, (100, 1)).float()

dataset = JetDataset(tokenized_data, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate model
model = JetAnomalyDetector(
    data=None,  # Will be set per batch
    vocab_size=8194,
    embedding_dim=256,
    max_sequence_len_per_jet=128
)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(5):
    for batch in loader:
        # Set model data for this batch
        model.data = {"jet1": batch["jet1"], "jet2": batch["jet2"]}
        logits = model()
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")