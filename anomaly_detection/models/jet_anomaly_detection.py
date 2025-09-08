# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# System imports
import os
import sys
import h5py
import logging
import numpy as np
import awkward as ak
from pathlib import Path
from datetime import datetime
from codenamegenerator import generate_codenames
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tqdm import tqdm

# Local Model imports
from gabbro.models.vqvae import VQVAELightning
from gabbro.models.vqvae import VQVAENormFormer
from gabbro.models.gpt_model import BackboneModel
from gabbro.data.loading import read_jetclass_file
from gabbro.utils.arrays import ak_select_and_preprocess, ak_pad, ak_to_np_stack
from gabbro.models.backbone import BackboneClassificationLightning
from torch.utils.data import TensorDataset, DataLoader

# Other imports
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_tokenizer")

def merge_tokenized_datasets(dataset1, dataset2):
    merged = {}
    for key in dataset1:
        merged[key] = np.concatenate([dataset1[key], dataset2[key]], axis=0)
    return merged

class TokenizationModule(nn.Module):
    """
    Tokenization module for jet data.
    """
    def __init__(
        self,
        data_path="/net/data_ttk/hreyes/LHCO/processed_jg/original/",
        save_dir="../checkpoints/",
        n_jets=5000,
        pad_length=128,
        batch_size=128,
        epochs=10,
        lr=1e-3,
        device=None,
        label_type="Signal",
        vqvae_ckpt_path="../checkpoints/vqvae_8192_tokens/model_ckpt.ckpt",
        use_pretrained_vqvae=True,
        generate_tokens=True,
        save_cache=True,
        verbose=False
    ):
        super().__init__()

        self.lr = lr
        self.n_jets = n_jets
        self.epochs = epochs
        self.verbose = verbose
        self.data_path = data_path
        self.pad_length = pad_length
        # maximum sequence length per jet used by the tokenizer
        self.max_sequence_len_per_jet = pad_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = os.path.join(save_dir, f"{generate_codenames()[0]}_{datetime.now().strftime('%Y%m%d')}")

        self.particle_features_tokenized = None  # Initialize to avoid AttributeError

        npz_path = os.path.join("cache", f"tokenized_{label_type.lower()}_data.npz")

        if verbose:
            print(f"\nProvided {label_type} dataset\n")

        if use_pretrained_vqvae:
            # ========== Frozen VQ-VAE Tokenizer (OmniJet-α style) ========== #
            print(f"\nLoading Pre-trained VQ-VAE model from {vqvae_ckpt_path}\n")
            self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt_path)
            self.vqvae_model.to(self.device).eval()
            cfg = OmegaConf.load(Path(vqvae_ckpt_path).parent / "config.yaml")
            pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common.feature_dict)
            print(f"\nLoaded VQ-VAE with {len(pp_dict)} features.\n")

            pp_dict_cuts = {
                    feat_name: {
                        criterion: pp_dict[feat_name].get(criterion)
                        for criterion in ["larger_than", "smaller_than"]
                    }
                    for feat_name in pp_dict
                }

            if self.verbose:
                print("\npp_dict:")
                for item in pp_dict:
                    print(item, pp_dict[item])

                print("\npp_dict_cuts:")
                for item in pp_dict_cuts:
                    print(item, pp_dict_cuts[item])

                print("\nModel:")
                print(self.vqvae_model)
                
        else: # training the VQ-VAE Tokenizer
            # ========== Train VQ-VAE Tokenizer (OmniJet-α style) ========== #
            os.makedirs(self.save_dir, exist_ok=True)

            print(f"Ready to Train Tokenizer. Tokenizer will save checkpoints to {self.save_dir}")

        if generate_tokens:
            # Check if tokenized data already exists, then load if available
            # Ask if user wants to use existing tokenized data
            if os.path.exists(npz_path):
                use_existing = input(f"Tokenized data found at {npz_path}. Do you want to use it? (y/n): ")
            else:
                use_existing = "n"
            if use_existing.lower() == "y" and os.path.exists(npz_path):
                print(f"\n\033[32mLoading tokenized data from {npz_path}\033[0m\n")
                with np.load(npz_path) as data:
                    self.particle_features_tokenized = {
                        "jet1": np.asarray(data["jet1"]),
                        "jet2": np.asarray(data["jet2"]),
                        "labels": np.asarray(data["labels"])
                }
                    
                # if verbose:   
                print("Dimensionality of Jet1:", self.particle_features_tokenized["jet1"].shape)
                print("Dimensionality of Jet2:", self.particle_features_tokenized["jet2"].shape)

            else:
                # Tokenize Input Data if cached data not found
                print("\nGenerating Tokens from Input Data\n")
                bg_file_path = os.path.join(data_path, "bg_N100_SR_extra.h5")
                sig_file_path = os.path.join(data_path, "sn_N100_SR.h5")
                #label_type = 'Background'
                requested_features = ["part_pt", "part_etarel", "part_phirel"]

                if label_type == 'Signal':
                    x_particles, _ = self.data_loader(
                        data_path=sig_file_path,
                        particle_features=requested_features,
                        jet_features=None,
                    )
                elif label_type == 'Background':
                    x_particles, _ = self.data_loader(
                        data_path=bg_file_path,
                        particle_features=requested_features,
                        jet_features=None,
                    )

                # data_loader returns a dict with keys 'jet1' and 'jet2'. Select jet1
                particle_features_ak_jet1 = x_particles["jet1"]
                particle_features_ak_jet2 = x_particles["jet2"]

                print("\nLoaded particle features (jet1): fields=", getattr(particle_features_ak_jet1, "fields", None))
                print("Loaded particle features (jet2): fields=", getattr(particle_features_ak_jet2, "fields", None))

                # If build_particle_matrix returned an unnamed numeric ragged array
                # (no fields), create a named-field awkward array as a fallback so
                # ak_select_and_preprocess can index by feature name.
                if not getattr(particle_features_ak_jet1, "fields", None) or "part_pt" not in particle_features_ak_jet1.fields:
                    print("particle_features_ak has no named fields, constructing named fields from columns")
                    # Each event in particle_features_ak is assumed to be an (n_const, n_feat) array-like
                    n_events = len(particle_features_ak_jet1)
                    ragged_fields = {}
                    for idx, name in enumerate(requested_features):
                        # build per-event lists of column idx
                        col = ak.Array([ (evt[:, idx] if len(evt) and evt.shape[1] > idx else []) for evt in particle_features_ak_jet1 ])
                        ragged_fields[name] = col
                    particle_features_ak_jet1 = ak.zip(ragged_fields)

                if not getattr(particle_features_ak_jet2, "fields", None) or "part_pt" not in particle_features_ak_jet2.fields:
                    print("particle_features_ak has no named fields, constructing named fields from columns")
                    # Each event in particle_features_ak is assumed to be an (n_const, n_feat) array-like
                    n_events = len(particle_features_ak_jet2)
                    ragged_fields = {}
                    for idx, name in enumerate(requested_features):
                        # build per-event lists of column idx
                        col = ak.Array([ (evt[:, idx] if len(evt) and evt.shape[1] > idx else []) for evt in particle_features_ak_jet2 ])
                        ragged_fields[name] = col
                    particle_features_ak_jet2 = ak.zip(ragged_fields)

                particle_features_ak_jet1 = ak_select_and_preprocess(particle_features_ak_jet1, pp_dict_cuts)[:, :128]
                particle_features_ak_jet2 = ak_select_and_preprocess(particle_features_ak_jet2, pp_dict_cuts)[:, :128]

                jet1 = self.tokenize_jets(particle_features_ak_jet1, pp_dict)
                jet1 = ak.pad_none(jet1, pad_length, axis=1)
                jet1 = ak.fill_none(jet1, 0)
                jet1_np = ak.to_numpy(jet1)

                jet2 = self.tokenize_jets(particle_features_ak_jet2, pp_dict)
                jet2 = ak.pad_none(jet2, pad_length, axis=1)
                jet2 = ak.fill_none(jet2, 0)
                jet2_np = ak.to_numpy(jet2)

                if label_type == 'Signal':
                    labels = np.ones(jet1_np.shape[0])
                elif label_type == 'Background':
                    labels = np.zeros(jet1_np.shape[0])
                else:
                    raise ValueError(f"Unknown label type: {label_type}")

                self.particle_features_tokenized = {"jet1": jet1_np, "jet2": jet2_np, "labels": labels}

                if save_cache:
                    # Save tokenized data to a compressed npz file
                    np.savez_compressed(npz_path,
                                        jet1=jet1_np,
                                        jet2=jet2_np,
                                        labels=labels)
                    print(f"\n\033[32mSaved tokenized data to {npz_path}\n\033[0m")

    def return_tokens(self):
        print("Returning tokenized data of length", len(self.particle_features_tokenized))
        return self.particle_features_tokenized
    
    def sample_jets(self, number):
        if not self.particle_features_tokenized:
            raise ValueError("No tokenized data available for sampling.")
        print("Sampled jets:")
        print("Jet1 shape:", self.particle_features_tokenized["jet1"].shape)
        print("Jet2 shape:", self.particle_features_tokenized["jet2"].shape)
        print("Label shape:", self.particle_features_tokenized["labels"].shape)
        N = self.particle_features_tokenized["jet1"].shape[0]
        idx = np.random.choice(N, size=number, replace=False)
        jet1_samples = self.particle_features_tokenized["jet1"][idx]
        jet2_samples = self.particle_features_tokenized["jet2"][idx]
        label_samples = self.particle_features_tokenized["labels"][idx]
        print("Sampled jets:")
        print("Jet1 samples:", jet1_samples.shape)
        print("Jet2 samples:", jet2_samples.shape)
        print("Label samples:", label_samples.shape)
        return {"jet1": jet1_samples, "jet2": jet2_samples, "labels": label_samples}

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

    def data_loader(
        self,
        data_path,
        particle_features=["part_pt", "part_eta", "part_phi", "part_energy"],
        jet_features=["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
        return_p4=False,
        n_load=None,
    ):
        """
        Load LHCO-style dijet HDF5 (bg_N100.h5 or sn_N100.h5).

        Inputs expected inside HDF5:
          - jet1/4mom, jet1/coords, jet1/features, jet1/mask
          - jet2/4mom, jet2/coords, jet2/features, jet2/mask
          - jet_coords: (2,4) -> (pt, eta, phi, m) for jet1 & jet2
          - jet_features: (7,) -> (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)

        Returns:
          x_particles: {"jet1": ak.Array, "jet2": ak.Array}
          x_jets: ak.Array with shape (n_events, 2, n_jet_features)
          p4 (optional): {"jet1": ak.Array, "jet2": ak.Array}
        """

        if self.verbose:
            self.inspect_data_file(data_path)

        with h5py.File(data_path, "r") as f:
            sl = slice(0, n_load) if n_load is not None else slice(None)

            # Read expected datasets using a progress bar to provide feedback for large files
            dataset_names = [
                "jet1/4mom",
                "jet1/coords",
                "jet1/features",
                "jet1/mask",
                "jet2/4mom",
                "jet2/coords",
                "jet2/features",
                "jet2/mask",
                "jet_coords",
                "jet_features",
            ]

            data = {}
            for name in tqdm(dataset_names, desc=f"Loading {os.path.basename(data_path)}", unit="ds"):
                # Use safe indexing with slice to avoid loading entire datasets if n_load is set
                data[name] = f[name][sl]

            j1_4mom   = data["jet1/4mom"]        # (N, 100, 4)  (e,px,py,pz)
            j1_coords = data["jet1/coords"]      # (N, 100, 2)  (eta,phi)
            j1_feats  = data["jet1/features"]    # (N, 100, 9)
            j1_mask   = data["jet1/mask"]        # (N, 100, 1)

            j2_4mom   = data["jet2/4mom"]
            j2_coords = data["jet2/coords"]
            j2_feats  = data["jet2/features"]
            j2_mask   = data["jet2/mask"]

            jet_coords = data["jet_coords"]      # (N, 2, 4) -> (pt, eta, phi, m)
            jet_feats7 = data["jet_features"]    # (N, 7) -> (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)

        # Build outputs using helpers
        x_particles = {
            "jet1": self.build_particle_matrix(j1_4mom, j1_coords, j1_feats, j1_mask, particle_features),
            "jet2": self.build_particle_matrix(j2_4mom, j2_coords, j2_feats, j2_mask, particle_features),
        }

        x_jets = self.build_jet_matrix(jet_coords, jet_feats7, j1_4mom, j2_4mom, j1_mask, j2_mask, jet_features)

        if return_p4:
            p4 = {
                "jet1": self.to_p4_ragged(j1_4mom, j1_mask),
                "jet2": self.to_p4_ragged(j2_4mom, j2_mask),
            }
            return x_particles, x_jets, p4

        return x_particles, x_jets
            
    def build_particle_matrix(self, jet_4mom, jet_coords, jet_feats, mask, requested_features):
        """
        Create a per-particle feature matrix (ragged per event) for a single jet.

        Supported feature names:
        part_pt, part_eta, part_phi, part_energy,
        part_log_pt, part_log_e, part_ptrel, part_erel, part_log_ptrel, part_log_erel, part_deltaR
        Aliases supported for compatibility: part_etarel -> part_eta, part_phirel -> part_phi
        """
        # 4-momentum (e, px, py, pz)
        e  = jet_4mom[..., 0]
        px = jet_4mom[..., 1]
        py = jet_4mom[..., 2]
        pz = jet_4mom[..., 3] 

        # Derived basics
        part_pt  = np.sqrt(px**2 + py**2)
        part_eta = jet_coords[..., 0]
        part_phi = jet_coords[..., 1]
        part_E   = e

        # HDF5 per-particle features columns:
        # (eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
        feature_bank = {
            "part_eta"      : part_eta,          # overrides with coords (same as feats[...,0])
            "part_phi"      : part_phi,          # overrides with coords (same as feats[...,1])
            # compatibility aliases (some codebases call these *_etarel / *_phirel)
            "part_etarel"   : part_eta,
            "part_phirel"   : part_phi,
            "part_log_pt"   : jet_feats[..., 2],
            "part_log_e"    : jet_feats[..., 3],
            "part_ptrel"    : jet_feats[..., 4],
            "part_erel"     : jet_feats[..., 5],
            "part_log_ptrel": jet_feats[..., 6],
            "part_log_erel" : jet_feats[..., 7],
            "part_deltaR"   : jet_feats[..., 8],
            # derived from 4mom:
            "part_pt"       : part_pt,
            "part_energy"   : part_E,
        }

        if not requested_features:
            return ak.Array([])

        # Normalize mask to shape (N, C) boolean
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.astype(bool)

        # Build per-feature ragged arrays and zip into a record array so fields
        # can be accessed by name (e.g., ak_array.part_pt) which is expected by
        # downstream utilities like ak_select_and_preprocess.
        ragged_fields = {}
        n_events = feature_bank[list(feature_bank.keys())[0]].shape[0]
        for name in tqdm(requested_features, desc="Building particle matrices"):
            if name not in feature_bank:
                raise ValueError(f"Unsupported particle feature: {name}")
            col = feature_bank[name]  # (N, C)
            # Create a list of per-event 1D arrays with masked entries removed
            ragged_fields[name] = ak.Array([col[i][mask[i]] for i in range(n_events)])

        out = ak.zip(ragged_fields)
        return out


    def build_jet_matrix(self, jet_coords, jet_feats7, j1_4mom, j2_4mom, j1_mask, j2_mask, requested_features):
        """
        Create per-event, per-jet features (2 jets per event).
        Supported jet feature names:
        jet_pt, jet_eta, jet_phi, jet_mass, jet_energy, jet_tau1, jet_tau2, jet_tau3, mjj

        Returns an Awkward Array of shape (N, 2, n_features).
        If 'mjj' is requested, it is replicated across both jets for shape consistency.
        """
        if j1_mask.ndim == 3 and j1_mask.shape[-1] == 1:
            j1_mask = j1_mask.squeeze(-1).astype(bool)
        if j2_mask.ndim == 3 and j2_mask.shape[-1] == 1:
            j2_mask = j2_mask.squeeze(-1).astype(bool)

        # jet_coords: (N, 2, 4) -> (pt, eta, phi, m)
        jet_pt   = jet_coords[..., 0]
        jet_eta  = jet_coords[..., 1]
        jet_phi  = jet_coords[..., 2]
        jet_mass = jet_coords[..., 3]

        # Per-jet energy (sum of constituent energies using mask)
        j1_E = (j1_4mom[..., 0] * j1_mask).sum(axis=1)  # (N,)
        j2_E = (j2_4mom[..., 0] * j2_mask).sum(axis=1)  # (N,)
        jet_energy = np.stack([j1_E, j2_E], axis=1)     # (N, 2)

        # tau1/2/3 for jet1 and jet2 from jet_features (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)
        tau1 = np.stack([jet_feats7[:, 0], jet_feats7[:, 3]], axis=1)  # (N,2)
        tau2 = np.stack([jet_feats7[:, 1], jet_feats7[:, 4]], axis=1)
        tau3 = np.stack([jet_feats7[:, 2], jet_feats7[:, 5]], axis=1)
        mjj  = jet_feats7[:, 6]  # (N,)

        bank = {
            "jet_pt"    : jet_pt,
            "jet_eta"   : jet_eta,
            "jet_phi"   : jet_phi,
            "jet_mass"  : jet_mass,
            "jet_energy": jet_energy,
            "jet_tau1"  : tau1,
            "jet_tau2"  : tau2,
            "jet_tau3"  : tau3,
            "mjj"       : mjj,   # event-level
        }

        if not requested_features:
            return ak.Array([])

        mats = []
        for name in tqdm(requested_features, desc="Building jet matrices"):
            if name == "mjj":
                # replicate mjj along jet axis: (N,) -> (N,2,1)
                mats.append(np.repeat(bank["mjj"][:, None, None], 2, axis=1))
            else:
                if name not in bank:
                    raise ValueError(f"Unsupported jet feature: {name}")
                mats.append(bank[name][..., None])  # (N,2,1)

        mat = np.concatenate(mats, axis=-1)  # (N,2,n_feat)
        return ak.Array(mat)


    def to_p4_ragged(self, jet_4mom, mask):
        """
        Convert per-particle 4-momentum (e,px,py,pz) to ragged Awkward arrays using mask.
        """
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1).astype(bool)
        arr = np.stack(
            [jet_4mom[..., 0], jet_4mom[..., 1], jet_4mom[..., 2], jet_4mom[..., 3]],
            axis=-1
        )  # (N, C, 4)
        out = ak.Array([arr[i][mask[i]] for i in range(arr.shape[0])])  # (n_const_i, 4)
        return out


    def inspect_data_file(self, h5_path: str) -> None:
        print(f"\n=== Inspecting: {h5_path} ===")
        with h5py.File(h5_path, "r") as f:
            print("Attributes:")
            for k, v in f.attrs.items():
                print(f"  - {k}: {v}")
            print("\nHierarchy:")
            def _printer(name):
                print(name)
            f.visit(_printer)
            def _safe_shape(name):
                try:
                    obj = f[name]
                    if isinstance(obj, h5py.Dataset):
                        return obj.shape, obj.dtype
                except Exception:
                    pass
                return None, None
            print("\nDataset shapes/dtypes:")
            for name in f:
                pass  # top-level only
            # Walk all datasets
            f.visititems(lambda name, obj: print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                        if isinstance(obj, h5py.Dataset) else None)
    
    def tokenize_jets(self, jets_ak, pp_dict, batch_size=512, pad_length=None):
        """
        Tokenizes the jet features using the VQ-VAE model.
        """
        pad_length = pad_length or self.max_sequence_len_per_jet
        with torch.no_grad():
            tokens = self.vqvae_model.tokenize_ak_array(
                ak_arr=jets_ak,
                pp_dict=pp_dict,
                batch_size=batch_size,
                pad_length=pad_length,
            )
        # ak.Array of shape (N_jets, L)
        return tokens
    
class JetDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, pad_length=128, pad_value=0):
        self.jet1 = torch.from_numpy(tokenized_data["jet1"]).long()
        self.jet2 = torch.from_numpy(tokenized_data["jet2"]).long()
        self.labels = torch.from_numpy(tokenized_data["labels"]).reshape(-1, 1)
    def __len__(self):
        return self.jet1.shape[0]
    def __getitem__(self, idx):
        return {
            "jet1": self.jet1[idx],
            "jet2": self.jet2[idx],
            "label": self.labels[idx]
        }


class JetAnomalyDetector(nn.Module):
    def __init__(
        self,
        data,
        vocab_size=8194,
        embedding_dim=256,
        max_sequence_len_per_jet=128,
        n_GPT_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        cls_token_id=0,
        device=None,
        verbose=False,
        # Downstream Transformer parameters
        downstream_num_layers=2,
        downstream_num_heads=2,
        jet_last_emb='linear',
        hlf_dim=5,
        dropout=0.1,
        num_layers_hlf=4,
        embedding_dim_hlf=256,
        sep_token_id=None,
        use_sep_token=False,
        use_hlf=False):

        super().__init__()

        self.use_hlf = use_hlf
        self.data = data
        self.verbose = verbose
        self.cls_token_id = cls_token_id
        self.embedding_dim = embedding_dim
        self.use_sep_token = use_sep_token
        self.jet_last_emb = jet_last_emb
        self.max_sequence_len_per_jet = max_sequence_len_per_jet
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sep_token_id = sep_token_id if sep_token_id is not None else cls_token_id

        

        # ========== OmniJet-α Backbone ========== #

        model_kwargs = {
            "embedding_dim": embedding_dim,
            "attention_dropout": attention_dropout,
            "vocab_size": vocab_size,
            "max_sequence_len": max_sequence_len_per_jet,
            "n_GPT_blocks": n_GPT_blocks,
            "n_heads": n_heads,
            "verbosity": True,
        }

        self.backbone = BackboneModel(**model_kwargs)
        self.backbone.to(self.device)


        # ========== Downstream Anomaly Detection Head ========== #
        # A second (small) Transformer that processes [CLS2, Jet1_repr, Jet2_repr]
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=downstream_num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.downstream_transformer = TransformerEncoder(encoder_layer, num_layers=downstream_num_layers)
        
        # A new CLS token for the downstream classification
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Final classification
        self.final_classifier = nn.Linear(embedding_dim, 1)

        # Criterion
        #self.criterion = nn.BCEWithLogitsLoss()
        

    def forward(self):

        # ========== Jet Data Pre-processing ========== #

        # bg_jet1,bg_jet2,bg_jet_coords = load_data(bgf)
        # sig_jet1,sig_jet2,sg_jet_coords =  load_data(sigf)

        # print(f"Using bg {bg_jet1.shape} from {bgf} and sig {sig_jet1.shape} from {sigf}")

        # jet1_data = np.concatenate((bg_jet1, sig_jet1), 0)
        # jet2_data = np.concatenate((bg_jet2, sig_jet2), 0)
        # jet_coords_data=np.concatenate((bg_jet_coords, sg_jet_coords), 0)
        
        
        # label = np.append(np.zeros(len(bg_jet1)), np.ones(len(sig_jet1)))
        
        # mask1 = jet1_data[:, :, 0] != 0
        # mask2 = jet2_data[:, :, 0] != 0

        # ========== OmniJet-α Backbone ========== #

        # Extract individual jets from provided data
        jet1 = self.data["jet1"]  # (N, L)
        jet2 = self.data["jet2"]  # (N, L)

        embeddings_jet1 = self.backbone(jet1)
        embeddings_jet2 = self.backbone(jet2)

        # ========== Downstream Anomaly Detection Head ========== #

        if self.use_hlf == False: # HLF = high-level features
            if self.use_sep_token:
                batch_size = embeddings_jet1.size(0)
                sep_token_expanded = self.sep_token.expand(batch_size, -1, -1)

                if self.jet_last_emb == 'mean':
                    # Stack: [jet1, SEP, jet2] along sequence dimension
                    jets_with_separator = torch.stack(
                        [embeddings_jet1, sep_token_expanded.squeeze(1), embeddings_jet2],
                        dim=1
                    )
                else:
                    # Concatenate along sequence dimension
                    jets_with_separator = torch.cat(
                        [embeddings_jet1, sep_token_expanded, embeddings_jet2],
                        dim=1
                    )
            else:
                if self.jet_last_emb == 'mean':
                    jets_with_separator = torch.stack([embeddings_jet1, embeddings_jet2], dim=1)
                else:
                    jets_with_separator = torch.cat([embeddings_jet1, embeddings_jet2], dim=1)

        # Insert a second CLS token at the front for downstream classification
        batch_size = embeddings_jet1.size(0)
        cls2_token_expanded = self.cls_token2.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]

        # Combine CLS2 + jets (and SEP if used)
        downstream_input_sequence = torch.cat([cls2_token_expanded, jets_with_separator], dim=1)

        # Pass through downstream Transformer for event-level representation
        downstream_output = self.downstream_transformer(downstream_input_sequence)

        # Extract CLS2 token representation for final classification (in index 0)
        event_representation = downstream_output[:, 0, :]  # [B, hidden_dim]

        # Final binary classification (e.g., signal vs background)
        logits = self.final_classifier(event_representation)  # [B, 1]

        # If labels are provided, return loss
        # if labels is not None:
        #     return logits, self.loss(logits, labels)
        
        return logits
    
    
    
