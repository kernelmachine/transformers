import json
import math
import numbers
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from torch import Size, Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint
from xformers.components.positional_embedding import RotaryEmbedding

from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import CausalLMOutputWithPast
from .configuration_openlm import OpenLMConfig


# from openclip
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs



class LayerNorm(nn.Module):
    # NOTE: taken from official pytorch implementation and modified
    # to allow revoval of gain and bias independently

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 0.00001,
        elementwise_gain: bool = True,
        elementwise_bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_gain = elementwise_gain
        self.elementwise_bias = elementwise_bias

        if self.elementwise_gain:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

        if self.elementwise_bias:
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_gain:
            with torch.no_grad():
                self.weight.fill_(1.0)

        if self.elementwise_bias:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_gain={elementwise_gain}, "
            "elementwise_bias={elementwise_bias}".format(**self.__dict__)
        )


class RmsNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps} ".format(**self.__dict__)


def get_norm_class(params):
    if params.model_norm == "default_layer_norm":
        return torch.nn.LayerNorm

    elif params.model_norm == "gain_only_layer_norm":
        return partial(LayerNorm, elementwise_gain=True, elementwise_bias=False)

    elif params.model_norm == "no_wb_layer_norm":
        return partial(LayerNorm, elementwise_gain=False, elementwise_bias=False)

    elif params.model_norm == "rms_norm":
        return RmsNorm

    else:
        raise ValueError(f"Unsupported model-norm: {params.model_norm}")


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


# args and default params follow llama (except with LayerNorm instead of RmsNorm)
@dataclass
class Params:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    norm_eps: float = 1e-5
    seq_len: int = 2048
    post_embed_norm: bool = False
    weight_tying: bool = False
    norm_type: nn.Module = nn.LayerNorm
    apply_qk_norm: bool = False


class RotaryWithCast(RotaryEmbedding):
    def forward(self, q, k, v):
        q, k = super().forward(q, k)
        return q.to(v.dtype), k.to(v.dtype), v


def xformers_attn(queries, keys, values, is_causal):
    mask = None
    if is_causal:
        mask = xops.LowerTriangularMask()
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=mask)


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.pos_embed = RotaryWithCast(self.head_dim, args.seq_len)
        self.attn_fn = xformers_attn
        self.apply_qk_norm = args.apply_qk_norm

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.out_proj.weight, std=std, a=-3 * std, b=3 * std
        )

        # initialize norm layers for queries and keys if needed
        self.q_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, is_causal=True):
        batchsize, seqlen, _ = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, seqlen, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, seqlen, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, seqlen, self.n_heads, self.head_dim)

        queries, keys, vals = self.pos_embed(queries, keys, vals)

        output = self.attn_fn(queries, keys, vals, is_causal=is_causal)

        output = output.view(batchsize, seqlen, -1)

        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args)

        # this follows llama / lit llama -- go to multiple of 256
        hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)

        self.feed_forward = xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False)
        self.layer_id = layer_id
        self.attention_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.attention.seq_len = args.seq_len

        # initialize weights trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(
            self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std
        )
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(hidden_dim)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x), is_causal=True)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.seq_len = params.seq_len
        self.post_embed_norm = (
            params.norm_type(
                params.dim,
                eps=params.norm_eps,
            )
            if params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = params.weight_tying

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(Block(layer_id, params))

        # get class for normalization layers
        self.norm = params.norm_type(
            params.dim,
            eps=params.norm_eps,
        )
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        if self.weight_tying:
            self.tok_embeddings.weight = self.output.weight
        self.grad_checkpointing = False

        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(
            self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, input):
        x = self.tok_embeddings(input)
        x = self.post_embed_norm(x)

        for layer in self.layers:
            if self.grad_checkpointing:
                x = checkpoint(layer, x)
            else:
                x = layer(x)

        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), x



def create_model(cfg):

    model_args = Params(
        dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        post_embed_norm=cfg.post_embed_norm,
        weight_tying=cfg.weight_tying,
    )
    model = Transformer(model_args)

    return model


class OpenLMModel(PreTrainedModel):
    config_class = OpenLMConfig

    def __init__(self, config):
        super().__init__(config)
        print(config)
        self.model = create_model(config)

    def forward(self, tokens):
        return self.model(tokens)


class OpenLMForCausalLM(OpenLMModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        #self.model = OpenLMModel(config)
        self.model = create_model(config)
        self.lm_head = None        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, OpenLlamaForCausalLM
        >>> model = OpenLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        logits, _ = self.model(input_ids)
        output = CausalLMOutputWithPast(
            logits=logits
        )
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


if __name__ == '__main__':
    openlm_config = OpenLMConfig.from_pretrained("utils/transformers/open_lm_config")
    print(OpenLMModel(openlm_config))