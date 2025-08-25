import os
from pathlib import Path
import argparse
import logging

import awkward as ak
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# repo imports (adjust PYTHONPATH if needed)
from gabbro.models.vqvae import VQVAENormFormer
from gabbro.data.loading import read_jetclass_file
from gabbro.utils.arrays import ak_select_and_preprocess, ak_pad, ak_to_np_stack

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_tokenizer")


class VQVAETokenization:
    """Encapsulate dataset prep and VQ-VAE tokenizer training.

    Usage:
        trainer = VQVAETokenization(data_path=..., save_dir=..., ...)
        trainer.train()
    """

    def __init__(
        self,
        data_path,
        save_dir="checkpoints/tokenizer",
        n_jets=5000,
        pad_length=128,
        batch_size=128,
        epochs=10,
        lr=1e-3,
        device=None,
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.n_jets = n_jets
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def build_pp_dict():
        return {
            "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "np.log", "inv_func": "np.exp"},
            "part_etarel": {"multiply_by": 3, "larger_than": -0.8, "smaller_than": 0.8},
            "part_phirel": {"multiply_by": 3, "larger_than": -0.8, "smaller_than": 0.8},
        }

    def load_and_preprocess(self, particle_features=None):
        if particle_features is None:
            particle_features = ["part_pt", "part_etarel", "part_phirel"]
        log.info("Reading data file: %s", self.data_path)
        part_features_ak, _, _ = read_jetclass_file(
            filepath=self.data_path,
            particle_features=particle_features,
            jet_features=None,
            labels=None,
            n_load=self.n_jets,
        )
        pp_dict = self.build_pp_dict()
        pp_dict_cuts = {
            feat: {crit: pp_dict[feat].get(crit) for crit in ["larger_than", "smaller_than"]}
            for feat in pp_dict
        }
        part_features_ak = ak_select_and_preprocess(part_features_ak, pp_dict_cuts)[:, : self.pad_length]
        part_features_padded, _mask = ak_pad(part_features_ak, maxlen=self.pad_length, return_mask=True)
        X = ak_to_np_stack(part_features_padded, names=list(pp_dict.keys()))
        return X, pp_dict

    def make_dataloader(self, X):
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    @staticmethod
    def parse_model_output(out):
        recon = None
        vq_loss = None
        info = {}
        if isinstance(out, (tuple, list)):
            if len(out) >= 1:
                recon = out[0]
            for item in out[1:]:
                if torch.is_tensor(item) and item.dim() == 0:
                    vq_loss = item
                    break
                if isinstance(item, dict):
                    info.update(item)
                    if "vq_loss" in item:
                        vq_loss = item["vq_loss"]
        elif isinstance(out, dict):
            recon = out.get("recon") or out.get("x_recon") or out.get("reconstruction")
            vq_loss = out.get("vq_loss") or out.get("commitment_loss")
            info.update(out)
        else:
            recon = out
        return recon, vq_loss, info

    def _build_model(self, input_dim):
        # Mirror config used in examples
        return VQVAENormFormer(
            input_dim=input_dim,
            hidden_dim=128,
            latent_dim=4,
            num_blocks=4,
            num_heads=8,
            alpha=10,
            vq_kwargs={
                "num_codes": 512,
                "beta": 0.9,
                "kmeans_init": True,
                "affine_lr": 2,
                "replace_freq": 100,
            },
        ).to(self.device)

    def save_checkpoint(self, model, pp_dict, epoch=None):
        if epoch is None:
            path = Path(self.save_dir) / "vqvae_final.pt"
        else:
            path = Path(self.save_dir) / f"vqvae_epoch{epoch}.pt"
        torch.save({"model_state_dict": model.state_dict(), "pp_dict": pp_dict}, path)
        log.info("Saved checkpoint: %s", path)

    def train(self):
        X, pp_dict = self.load_and_preprocess()
        log.info("Loaded data shape: %s", X.shape)
        loader = self.make_dataloader(X)

        model = self._build_model(input_dim=X.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
        recon_criterion = nn.MSELoss(reduction="mean")

        log.info("Starting training on device: %s", self.device)
        for epoch in range(1, self.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                out = model(x)
                recon, vq_loss, info = self.parse_model_output(out)
                if recon is None:
                    raise RuntimeError("Model forward did not return a reconstruction tensor.")
                if recon.shape != x.shape:
                    recon = recon.view_as(x)
                recon_loss = recon_criterion(recon, x)
                if vq_loss is None:
                    vq_loss = info.get("vq_loss")
                loss = recon_loss + vq_loss if vq_loss is not None else recon_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            epoch_loss = running_loss / X.shape[0]
            log.info("Epoch %d/%d  Loss: %.6f", epoch, self.epochs, epoch_loss)
            self.save_checkpoint(model, pp_dict, epoch=epoch)

        self.save_checkpoint(model, pp_dict, epoch=None)
        log.info("Training finished. Final model saved to %s", self.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE tokenizer directly (no Hydra).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to a JetClass ROOT file (or compatible) used by read_jetclass_file.")
    parser.add_argument("--n_jets", type=int, default=5000)
    parser.add_argument("--pad_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints/tokenizer")
    args = parser.parse_args()

    trainer = VQVAETokenization(
        data_path=args.data_path,
        save_dir=args.save_dir,
        n_jets=args.n_jets,
        pad_length=args.pad_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    trainer.train()