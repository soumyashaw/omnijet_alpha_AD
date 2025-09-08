# PyTorch imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
from models.utils import plot_roc_curve, get_anomaly_scores
np.set_printoptions(threshold=np.inf)

# Anomaly Detection class imports
import models.jet_anomaly_detection as jet_ad
from models import TokenizationModule, JetAnomalyDetector, JetDataset

# Example training setup
def train():
    # Create model
    tokenized_data_signal = TokenizationModule(use_pretrained_vqvae=True, label_type="Signal").sample_jets(75000)
    tokenized_data_background = TokenizationModule(use_pretrained_vqvae=True, label_type="Background").sample_jets(350000)

    # Split Signal Data into 50k and 25k split
    tokenized_data_signal_50k = {key: value[:50000] for key, value in tokenized_data_signal.items()}  # 50k signal data
    tokenized_data_signal_25k = {key: value[50000:] for key, value in tokenized_data_signal.items()}  # 25k signal data

    # Split Background Data into 200k and 150k split
    tokenized_data_background_200k = {key: value[:200000] for key, value in tokenized_data_background.items()}  # 200k background data
    tokenized_data_background_150k = {key: value[200000:] for key, value in tokenized_data_background.items()}  # 150k background data

    train_data = jet_ad.merge_tokenized_datasets(tokenized_data_signal_25k, tokenized_data_background_200k)
    test_data = jet_ad.merge_tokenized_datasets(tokenized_data_signal_50k, tokenized_data_background_150k)

    print("train_data.shape", train_data["jet1"].shape, train_data["jet2"].shape, train_data["labels"].shape)
    print("test_data.shape", test_data["jet1"].shape, test_data["jet2"].shape, test_data["labels"].shape)

    # Split data into train and test (80-20)
    # total_samples = len(tokenized_data["labels"])
    # train_size = int(0.8 * total_samples)
    # indices = np.arange(total_samples)
    # np.random.shuffle(indices)
    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:]

    # train_data = {
    #     "jet1": tokenized_data["jet1"][train_indices],
    #     "jet2": tokenized_data["jet2"][train_indices],
    #     "labels": tokenized_data["labels"][train_indices]
    # }
    # test_data = {
    #     "jet1": tokenized_data["jet1"][test_indices],
    #     "jet2": tokenized_data["jet2"][test_indices],
    #     "labels": tokenized_data["labels"][test_indices]
    # }

    dataset_train = JetDataset(train_data)
    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

    dataset_test = JetDataset(test_data)
    loader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JetAnomalyDetector(data=None, verbose=True).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(3):
        for batch in tqdm(loader_train, desc=f"Epoch {epoch+1}"):
            # Set model data for this batch
            model.data = {"jet1": batch["jet1"].to(device), "jet2": batch["jet2"].to(device)}
            logits = model()
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader_test:
            model.data = {"jet1": batch["jet1"].to(device), "jet2": batch["jet2"].to(device)}
            logits = model()
            loss = criterion(logits, batch["label"].to(device))
            test_loss += loss.item() * batch["label"].size(0)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == batch["label"].to(device)).sum().item()
            total += batch["label"].size(0)
            # Collect predictions and true labels
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    test_loss /= total
    accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Concatenate all predictions and labels
    y_preds = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Write the predicted and true labels to a text file
    output_file = "/home/home3/institut_thp/soshaw/omnijet_alpha_AD/anomaly_detection/results/predictions.txt"
    with open(output_file, 'w') as f:
        f.write("TrueLabel\tPredictedScore\n")
        for true_label, pred_score in zip(y_true, y_preds):
            f.write(f"{true_label}\t{pred_score[0]}\n")

    auc = get_anomaly_scores(y_true, y_preds)

    # Save the last checkpoint
    checkpoint_path = "/home/home3/institut_thp/soshaw/omnijet_alpha_AD/anomaly_detection/checkpoints/last_checkpoint.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'test_loss': test_loss,
        'accuracy': accuracy
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return y_preds, y_true

if __name__ == "__main__":
    train()