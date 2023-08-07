from ...configuration_utils import PretrainedConfig
from typing import List
import logging

logger = logging.getLogger(__name__)

OPEN_LM_1B_CONFIG = {
    "hidden_dim": 2048,
    "n_layers": 24,
    "n_heads": 16,
    "seq_len": 2048,
    "vocab_size": 50432,
    "post_embed_norm": False,
    "weight_tying": False
}

class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"
    def __init__(
        self,
        hidden_dim: int = 768, 
        n_layers: int = 12,
        n_heads: int = 12,
        seq_len: int = 2048,
        vocab_size: int = 50304,
        post_embed_norm: bool = False,
        weight_tying: bool = False,
        **kwargs
    ):
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.post_embed_norm = post_embed_norm
        self.weight_tying = weight_tying
        super().__init__(**kwargs)



if __name__ == '__main__':
    config = {
        "hidden_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "seq_len": 2048,
        "vocab_size": 50304,
        "pre_ln": False,
        "pos_embed_type": "rope",
        "weight_tying": False,
        "attn_type": "xformers"
    }
    openlm_config = OpenLMConfig(config)
    openlm_config.save_pretrained("open_lm_config")