# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# System imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local Model imports
from gabbro.models.vqvae import VQVAELightning
from gabbro.models.gpt_model import BackboneModel
from gabbro.data.loading import read_jetclass_file
from gabbro.utils.arrays import ak_select_and_preprocess
from gabbro.models.backbone import BackboneClassificationLightning

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
        self.cls_token_id = cls_token_id
        self.embedding_dim = embedding_dim
        self.max_sequence_len_per_jet = max_sequence_len_per_jet
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sep_token_id = sep_token_id if sep_token_id is not None else cls_token_id
        

        # ========== Frozen VQ-VAE Tokenizer (OmniJet-α style) ========== #
        # Load VQ-VAE model
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

        embeddings_jet1 = self.backbone(self.particle_features_ak_tokenized, mask1)
        embeddings_jet2 = self.backbone(self.particle_features_ak_tokenized, mask2)

        # ========== Downstream Anomaly Detection Head ========== #

        if self.use_hlf == False: # HLF = high-level features
            if self.use_sep_token == 'True':
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
    
