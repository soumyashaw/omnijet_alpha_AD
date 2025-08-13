import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import lightning as L
import numpy as np
import torch.optim as optim
import torch.nn as nn
from gabbro.models.vqvae import VQVAELightning
from gabbro.models.backbone import BackboneClassificationLightning

# ---------------------------------------------------------------------------------- #
# NOTE: The class below (JetAnomalyDetector) has shortcomings for anomaly detection:  #
#  - It feeds (B, T, F) token-wise features directly into a Linear(feature_dim,1),   #
#    producing (B, T) scores that are (silently) broadcast against (B,) labels.       #
#  - It does not combine TWO jets; it treats each jet independently.                 #
#  - It bypasses the BackboneClassificationLightning head logic.                     #
#                                                                                    #
# We keep it for backward compatibility, but add a corrected implementation further  #
# below: JetPairAnomalyDetector, which:                                               #
#   * Accepts pairs of jets.                                                         #
#   * Builds a combined token sequence: [CLS] jet1_tokens [SEP] jet2_tokens          #
#   * Uses BackboneClassificationLightning with a single output node (n_out_nodes=1) #
#     so the existing classification head yields one anomaly score per jet pair.     #
#   * Properly aggregates sequence embeddings (summation head) instead of per-token  #
#     regression.                                                                    #
# ---------------------------------------------------------------------------------- #

class JetAnomalyDetector1:
    def __init__(
        self,
        vqvae_ckpt_path="../checkpoints/vqvae_8192_tokens/model_ckpt.ckpt",
        vocab_size=8194,
        embedding_dim=256,
        max_sequence_len=128,
        n_GPT_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load and freeze the tokenization model
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt_path)
        self.vqvae_model = self.vqvae_model.to(self.device)
        self.vqvae_model.eval()
        for p in self.vqvae_model.parameters():
            p.requires_grad = False

        # Build transformer backbone (without classification head)
        model_kwargs = {
            "embedding_dim": embedding_dim,
            "attention_dropout": attention_dropout,
            "vocab_size": vocab_size,
            "max_sequence_len": max_sequence_len,
            "n_GPT_blocks": n_GPT_blocks,
            "n_heads": n_heads,
            "n_out_nodes": 2,  # Placeholder, not used
            "verbosity": False,
        }
        dummy_param = torch.nn.Parameter(torch.empty(0))
        optimizer = optim.AdamW([dummy_param], lr=0.001, weight_decay=0)
        self.backbone = BackboneClassificationLightning(
            optimizer=optimizer,
            scheduler=None,
            class_head_type="summation",
            model_kwargs=model_kwargs,
        )
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()  # We'll set to train() in training

        # Anomaly score head (trainable)
        # We'll initialize after seeing feature dim
        self.anomaly_head = None

    def tokenize_jets(self, part_features_ak, pp_dict, batch_size=512, pad_length=128):
        with torch.no_grad():
            tokens = self.vqvae_model.tokenize_ak_array(
                ak_arr=part_features_ak,
                pp_dict=pp_dict,
                batch_size=batch_size,
                pad_length=pad_length,
            )
        return tokens

    def backbone_features(self, tokenized_jets):
        if not isinstance(tokenized_jets, torch.Tensor):
            tokenized_jets = torch.tensor(tokenized_jets, dtype=torch.long)
        tokenized_jets = tokenized_jets.to(self.device)
        mask = torch.ones(tokenized_jets.shape, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            features = self.backbone.module(tokenized_jets, mask)
        return features

    def setup_anomaly_head(self, feature_dim):
        self.anomaly_head = nn.Linear(feature_dim, 1).to(self.device)

    def predict_anomaly_score(self, features):
        if self.anomaly_head is None:
            self.setup_anomaly_head(features.shape[-1])
        # Expect features of shape (B, T, F); aggregate over tokens first.
        if features.dim() == 3:
            features_agg = features.mean(dim=1)
        else:
            features_agg = features
        scores = self.anomaly_head(features_agg).squeeze(-1)
        return scores

    def train(
        self,
        tokenized_jets_train,
        anomaly_labels_train,
        epochs=10,
        lr=1e-3,
        batch_size=64,
    ):
        self.backbone.eval()  # backbone frozen
        for p in self.backbone.parameters():
            p.requires_grad = False
        if not isinstance(tokenized_jets_train, torch.Tensor):
            tokenized_jets_train = torch.tensor(tokenized_jets_train, dtype=torch.long)
        tokenized_jets_train = tokenized_jets_train.to(self.device)
        anomaly_labels_train = torch.tensor(anomaly_labels_train, dtype=torch.float32).to(self.device)

        # Get features for all training data (backbone is frozen)
        with torch.no_grad():
            features = self.backbone_features(tokenized_jets_train)
        if self.anomaly_head is None:
            self.setup_anomaly_head(features.shape[-1])

        self.anomaly_head.train()
        optimizer = optim.AdamW(self.anomaly_head.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(features, anomaly_labels_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                # batch_features shape: (B, T, F)
                preds = self.predict_anomaly_score(batch_features)
                loss = criterion(preds, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_features.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def test(self, tokenized_jets_test, anomaly_labels_test=None):
        self.backbone.eval()
        self.anomaly_head.eval()
        if not isinstance(tokenized_jets_test, torch.Tensor):
            tokenized_jets_test = torch.tensor(tokenized_jets_test, dtype=torch.long)
        tokenized_jets_test = tokenized_jets_test.to(self.device)
        with torch.no_grad():
            features = self.backbone_features(tokenized_jets_test)
            scores = self.predict_anomaly_score(features)
        if anomaly_labels_test is not None:
            anomaly_labels_test = torch.tensor(anomaly_labels_test, dtype=torch.float32).to(self.device)
            mse = nn.MSELoss()(scores, anomaly_labels_test)
            print(f"Test MSE: {mse.item():.4f}")
        return scores


# ---------------------------------------------------------------------------------- #
# New, recommended implementation for pair-wise anomaly detection using backbone.     #
# ---------------------------------------------------------------------------------- #
class JetPairAnomalyDetector2(nn.Module):
    """Pair-wise anomaly detector.

    Pipeline:
      1. Frozen VQ-VAE tokenizes each jet independently.
      2. Build paired sequences: [CLS] jet1_tokens [SEP] jet2_tokens (truncate/pad).
      3. Pass combined sequence to backbone (BackboneClassificationLightning) with
         a *single* output node (n_out_nodes=1) so the head outputs anomaly score.
    """

    def __init__(
        self,
        vqvae_ckpt_path="../checkpoints/vqvae_8192_tokens/model_ckpt.ckpt",
        vocab_size=8194,
        embedding_dim=256,
        max_sequence_len_per_jet=128,
        n_GPT_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        cls_token_id=0,
        sep_token_id=None,  # if None, will reuse cls_token_id (can customize)
        device=None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id if sep_token_id is not None else cls_token_id
        self.max_sequence_len_per_jet = max_sequence_len_per_jet

        # 1) Load & freeze tokenizer
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt_path)
        self.vqvae_model.to(self.device).eval()
        for p in self.vqvae_model.parameters():
            p.requires_grad = False

        # 2) Backbone with single output node
        model_kwargs = {
            "embedding_dim": embedding_dim,
            "attention_dropout": attention_dropout,
            "vocab_size": vocab_size,
            # Combined length: 1 (CLS) + L + 1 (SEP) + L  <= Provide to backbone
            "max_sequence_len": 1 + max_sequence_len_per_jet + 1 + max_sequence_len_per_jet,
            "n_GPT_blocks": n_GPT_blocks,
            "n_heads": n_heads,
            "n_out_nodes": 1,  # single anomaly score
            "verbosity": False,
        }
        dummy_param = torch.nn.Parameter(torch.empty(0))
        optimizer = optim.AdamW([dummy_param], lr=1e-3, weight_decay=0)
        self.backbone = BackboneClassificationLightning(
            optimizer=optimizer,
            scheduler=None,
            class_head_type="summation",  # sum embeddings --> anomaly score
            model_kwargs=model_kwargs,
        ).to(self.device)
        # Optionally freeze backbone (set requires_grad=False) if only head training desired
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

    # ---------------- Tokenization utilities ---------------- #
    def tokenize_jets(self, jets_ak, pp_dict, batch_size=512, pad_length=None):
        pad_length = pad_length or self.max_sequence_len_per_jet
        with torch.no_grad():
            tokens = self.vqvae_model.tokenize_ak_array(
                ak_arr=jets_ak,
                pp_dict=pp_dict,
                batch_size=batch_size,
                pad_length=pad_length,
            )
        return tokens  # ak.Array of shape (N_jets, L)

    # ---------------- Pair building ---------------- #
    def build_paired_tensor(self, jet_tokens_1, jet_tokens_2):
        """Create combined sequence [CLS] jet1 [SEP] jet2.

        jet_tokens_* : torch.LongTensor or np.ndarray shape (B, L)
        Returns: (B, L_combined), mask (B, L_combined)
        """
        if not isinstance(jet_tokens_1, torch.Tensor):
            jet_tokens_1 = torch.tensor(jet_tokens_1, dtype=torch.long)
        if not isinstance(jet_tokens_2, torch.Tensor):
            jet_tokens_2 = torch.tensor(jet_tokens_2, dtype=torch.long)
        assert jet_tokens_1.shape == jet_tokens_2.shape
        B, L = jet_tokens_1.shape
        cls_col = torch.full((B, 1), self.cls_token_id, device=jet_tokens_1.device)
        sep_col = torch.full((B, 1), self.sep_token_id, device=jet_tokens_1.device)
        combined = torch.cat([cls_col, jet_tokens_1.to(cls_col.device), sep_col, jet_tokens_2.to(cls_col.device)], dim=1)
        mask = torch.ones_like(combined, dtype=torch.float32)
        return combined.to(self.device), mask.to(self.device)

    # ---------------- Forward / scoring ---------------- #
    def forward_pairs(self, jet_tokens_1, jet_tokens_2):
        self.backbone.eval()  # backbone inference mode (unless fine-tuning)
        X, mask = self.build_paired_tensor(jet_tokens_1, jet_tokens_2)
        with torch.no_grad():
            logits = self.backbone(X, mask)  # (B, 1)
        return logits.squeeze(-1)

    # ---------------- Training the head/backbone ---------------- #
    def fit(self, jet_tokens_1, jet_tokens_2, labels, epochs=5, batch_size=32, lr=1e-4, finetune_backbone=False):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        if not finetune_backbone:
            for p in self.backbone.module.parameters():
                p.requires_grad = False
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(jet_tokens_1, dtype=torch.long),
            torch.tensor(jet_tokens_2, dtype=torch.long),
            labels,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Only optimize parameters with requires_grad True
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.backbone.train()
        for epoch in range(epochs):
            total = 0.0
            for jets_a, jets_b, y in loader:
                jets_a, jets_b, y = jets_a.to(self.device), jets_b.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                X, mask = self.build_paired_tensor(jets_a, jets_b)
                logits = self.backbone(X, mask).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total += loss.item() * y.size(0)
            print(f"Epoch {epoch+1}/{epochs}  Loss: {total/len(dataset):.4f}")

    def predict(self, jet_tokens_1, jet_tokens_2):
        return torch.sigmoid(self.forward_pairs(jet_tokens_1, jet_tokens_2)).detach().cpu()


# ---------------------------------------------------------------------------------- #
# LightningModule version for integration into existing training loops.              #
# Expects dataloader batches with keys: {"jet_tokens_a", "jet_tokens_b", "label"}.   #
# Tokenization (VQ-VAE) is assumed precomputed; class can optionally load tokenizer. #
# ---------------------------------------------------------------------------------- #
class JetPairAnomalyLightning(L.LightningModule):
    def __init__(
        self,
        optimizer_init=None,
        scheduler_init=None,
        vqvae_ckpt_path=None,
        vocab_size=8194,
        embedding_dim=256,
        max_sequence_len_per_jet=128,
        n_GPT_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        cls_token_id=0,
        sep_token_id=None,
        finetune_backbone=False,
        freeze_backbone=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id if sep_token_id is not None else cls_token_id
        self.max_sequence_len_per_jet = max_sequence_len_per_jet
        self.finetune_backbone = finetune_backbone
        self.freeze_backbone_flag = freeze_backbone and not finetune_backbone

        # Optional tokenizer (unused in forward if tokens already prepared)
        if vqvae_ckpt_path is not None:
            self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt_path)
            self.vqvae_model.eval()
            for p in self.vqvae_model.parameters():
                p.requires_grad = False
        else:
            self.vqvae_model = None

        model_kwargs = {
            "embedding_dim": embedding_dim,
            "attention_dropout": attention_dropout,
            "vocab_size": vocab_size,
            "max_sequence_len": 1 + max_sequence_len_per_jet + 1 + max_sequence_len_per_jet,
            "n_GPT_blocks": n_GPT_blocks,
            "n_heads": n_heads,
            "n_out_nodes": 1,
            "verbosity": False,
        }
        # dummy optimizer placeholder for backbone lightning module
        dummy_param = torch.nn.Parameter(torch.empty(0))
        backbone_optimizer = optim.AdamW([dummy_param], lr=1e-3)
        self.backbone = BackboneClassificationLightning(
            optimizer=backbone_optimizer,
            scheduler=None,
            class_head_type="summation",
            model_kwargs=model_kwargs,
        )

        # Freeze only the transformer backbone (self.backbone.module) if requested, keep head trainable
        if self.freeze_backbone_flag:
            for p in self.backbone.module.parameters():
                p.requires_grad = False
            for p in self.backbone.head.parameters():
                p.requires_grad = True  # ensure classification head can still learn

        self.criterion = nn.BCEWithLogitsLoss()
        self._optimizer_init = optimizer_init or (lambda params: optim.AdamW(params, lr=1e-4))
        self._scheduler_init = scheduler_init  # callable taking optimizer -> scheduler or None

    # ---------------- util ---------------- #
    def build_paired_tensor(self, jet_tokens_1, jet_tokens_2):
        if not isinstance(jet_tokens_1, torch.Tensor):
            jet_tokens_1 = torch.tensor(jet_tokens_1, dtype=torch.long)
        if not isinstance(jet_tokens_2, torch.Tensor):
            jet_tokens_2 = torch.tensor(jet_tokens_2, dtype=torch.long)
        assert jet_tokens_1.shape == jet_tokens_2.shape, "Jet pair tensors must have equal shape"
        B, L = jet_tokens_1.shape
        if L > self.max_sequence_len_per_jet:
            jet_tokens_1 = jet_tokens_1[:, : self.max_sequence_len_per_jet]
            jet_tokens_2 = jet_tokens_2[:, : self.max_sequence_len_per_jet]
            L = self.max_sequence_len_per_jet
        cls_col = torch.full((B, 1), self.cls_token_id, device=jet_tokens_1.device, dtype=torch.long)
        sep_col = torch.full((B, 1), self.sep_token_id, device=jet_tokens_1.device, dtype=torch.long)
        combined = torch.cat([cls_col, jet_tokens_1.to(cls_col.device), sep_col, jet_tokens_2.to(cls_col.device)], dim=1)
        # full attention mask (no exclusion to avoid -inf rows)
        attn_mask = torch.ones(combined.size(0), combined.size(1), device=combined.device, dtype=torch.float32)
        # pooling mask excludes CLS and SEP for aggregation
        pooling_mask = attn_mask.clone()
        pooling_mask[:, 0] = 0
        pooling_mask[:, 1 + L] = 0
        return combined.to(self.device), attn_mask.to(self.device), pooling_mask.to(self.device)

    # ---------------- forward ---------------- #
    def forward(self, jet_tokens_1, jet_tokens_2):
        X, attn_mask, pooling_mask = self.build_paired_tensor(jet_tokens_1, jet_tokens_2)
        embeddings = self.backbone.module(X, attn_mask)
        logits = self.backbone.head(embeddings, pooling_mask).squeeze(-1)
        if torch.isnan(logits).any():  # debug aid
            raise RuntimeError("NaN detected in logits; check input tokens and masks.")
        return logits

    # ---------------- lightning steps ---------------- #
    def training_step(self, batch, batch_idx):
        logits = self(batch["jet_tokens_a"], batch["jet_tokens_b"])
        labels = batch["label"].float().to(logits.device)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["jet_tokens_a"], batch["jet_tokens_b"])
        labels = batch["label"].float().to(logits.device)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        if "val_scores" not in self.trainer.callback_metrics:
            # nothing; metrics aggregator can be added later
            pass
        return {"val_loss": loss, "preds": preds.detach(), "labels": labels.detach()}

    def test_step(self, batch, batch_idx):
        logits = self(batch["jet_tokens_a"], batch["jet_tokens_b"])
        labels = batch.get("label")
        loss = None
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss = self.criterion(logits, labels)
            self.log("test_loss", loss, prog_bar=True)
        scores = torch.sigmoid(logits)
        return {"scores": scores.detach(), "loss": loss.detach() if loss is not None else None}

    def configure_optimizers(self):
        # Only optimize unfrozen parameters (e.g. classification head and optionally backbone)
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self._optimizer_init(params)
        if self._scheduler_init is not None:
            scheduler = self._scheduler_init(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
    # ---------------- convenience non-Lightning training (optional) ---------------- #
    def fit(self, jet_tokens_a, jet_tokens_b, labels, epochs=5, batch_size=32, lr=None):
        """Manual training loop for quick experiments without a Trainer.

        Args:
            jet_tokens_a, jet_tokens_b: array-like (N, L)
            labels: array-like (N,) binary (0/1)
            epochs: int
            batch_size: int
            lr: override learning rate (else use optimizer_init default)
        Note: Uses BCEWithLogitsLoss. Respects frozen backbone parameters.
        """
        # Build dataset
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(jet_tokens_a, dtype=torch.long),
            torch.tensor(jet_tokens_b, dtype=torch.long),
            labels,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Create optimizer (only trainable params)
        params = [p for p in self.parameters() if p.requires_grad]
        # Fallback: if everything was frozen (e.g. earlier version froze entire backbone), unfreeze head
        if len(params) == 0:
            for p in self.backbone.head.parameters():
                p.requires_grad = True
            params = [p for p in self.parameters() if p.requires_grad]
            if len(params) == 0:
                raise ValueError("No trainable parameters found. Set freeze_backbone=False when constructing JetPairAnomalyLightning or ensure head params are unfrozen.")
        if lr is not None:
            optimizer = optim.AdamW(params, lr=lr)
        else:
            optimizer = self._optimizer_init(params)
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for a, b, y in loader:
                a, b, y = a.to(self.device), b.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(a, b)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
            print(f"[ManualFit] Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(ds):.4f}")
        self.eval()

    @torch.no_grad()
    def predict(self, jet_tokens_a, jet_tokens_b, batch_size=256, to_numpy=True):
        """Batch inference returning sigmoid scores."""
        if not isinstance(jet_tokens_a, torch.Tensor):
            jet_tokens_a = torch.tensor(jet_tokens_a, dtype=torch.long)
        if not isinstance(jet_tokens_b, torch.Tensor):
            jet_tokens_b = torch.tensor(jet_tokens_b, dtype=torch.long)
        assert jet_tokens_a.shape == jet_tokens_b.shape
        self.eval()
        scores = []
        for start in range(0, jet_tokens_a.size(0), batch_size):
            end = start + batch_size
            a = jet_tokens_a[start:end].to(self.device)
            b = jet_tokens_b[start:end].to(self.device)
            logits = self.forward(a, b)
            scores.append(torch.sigmoid(logits).cpu())
        out = torch.cat(scores, dim=0)
        return out.numpy() if to_numpy else out

# Example usage:
if __name__ == "__main__":
    # Example usage of new JetPairAnomalyDetector with random tokens
    B = 4
    L = 128
    jet_tokens_a = np.random.randint(0, 8194, size=(B, L))
    jet_tokens_b = np.random.randint(0, 8194, size=(B, L))
    labels = np.random.randint(0, 2, size=(B,)).astype(float)

    detector = JetPairAnomalyLightning()
    detector.fit(jet_tokens_a, jet_tokens_b, labels, epochs=2, batch_size=2, lr=1e-4)
    preds = detector.predict(jet_tokens_a, jet_tokens_b)
    print("Predicted anomaly scores (sigmoid):", preds)