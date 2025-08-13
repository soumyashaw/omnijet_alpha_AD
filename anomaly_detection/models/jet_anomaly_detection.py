# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# System imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local Model imports
from gabbro.models.vqvae import VQVAELightning
from gabbro.data.loading import read_jetclass_file
from gabbro.utils.arrays import ak_select_and_preprocess

# Other imports
from omegaconf import OmegaConf

class JetAnomalyDetector(nn.Module):
    def __init__(self,
        vqvae_ckpt_path="../checkpoints/vqvae_8192_tokens/model_ckpt.ckpt",
        vocab_size=8194,
        embedding_dim=256,
        max_sequence_len_per_jet=128,
        n_GPT_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        cls_token_id=0,
        sep_token_id=None,
        device=None,
        verbose=False):

        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id if sep_token_id is not None else cls_token_id
        self.max_sequence_len_per_jet = max_sequence_len_per_jet

        # Load VQ-VAE model (OmniJet-α style)
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt_path)
        self.vqvae_model.to(self.device).eval()
        cfg = OmegaConf.load(Path(vqvae_ckpt_path).parent / "config.yaml")
        pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common.feature_dict)

        pp_dict_cuts = {
                feat_name: {
                    criterion: pp_dict[feat_name].get(criterion)
                    for criterion in ["larger_than", "smaller_than"]
                }
                for feat_name in pp_dict
            }

        if verbose:
            print("\npp_dict:")
            for item in pp_dict:
                print(item, pp_dict[item])

            print("\npp_dict_cuts:")
            for item in pp_dict_cuts:
                print(item, pp_dict_cuts[item])

            print("\nModel:")
            print(self.vqvae_model)

        jetclass_file_path = (
            "/.automount/home/home__home3/institut_thp/soshaw/test_20M/ZJetsToNuNu_100.root"
        )

        particle_features_ak, _, _ = read_jetclass_file(
            filepath=jetclass_file_path,
            particle_features=["part_pt", "part_etarel", "part_phirel"],
            jet_features=None,
            labels=None,
            n_load=100_000,
        )

        particle_features_ak = ak_select_and_preprocess(particle_features_ak, pp_dict_cuts)[:, :128]

        particle_features_ak_tokenized = self.tokenize_jets(particle_features_ak, pp_dict)

        print(particle_features_ak_tokenized)

        # Define layers
        #self.fc1 = nn.Linear(input_size, hidden_size)  # first fully connected layer
        #self.fc2 = nn.Linear(hidden_size, output_size) # second fully connected layer
        
        # (Optional) define other layers: Conv2d, LSTM, BatchNorm, Dropout, etc.

    def forward(self, x):
        pass



        # Define the forward pass
        #x = self.fc1(x)         # input → first layer
        #x = F.relu(x)           # activation
        #x = self.fc2(x)         # → second layer
        #return x                # output prediction
    
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