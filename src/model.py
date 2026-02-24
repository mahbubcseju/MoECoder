import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.modeling_utils import load_sharded_checkpoint
except ImportError:  # pragma: no cover - older transformers versions
    load_sharded_checkpoint = None


MOE_METADATA_FILENAME = "moe_config.json"


class MoEMLP(nn.Module):
    def __init__(self, dense_mlp, hidden_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # inside MoEMLP.__init__
        ref = next(dense_mlp.parameters())
        self.router = nn.Linear(hidden_size, num_experts, bias=False).to(
            device=ref.device, dtype=ref.dtype
        )
        self.experts = nn.ModuleList([copy.deepcopy(dense_mlp) for _ in range(num_experts)])
        self.last_aux_loss = None

    # def _compute_aux_loss(self, router_logits, topk_indices):
    #     router_probs = torch.softmax(router_logits.float(), dim=-1)
    #     importance = router_probs.mean(dim=(0, 1))
    #     hard_assign = F.one_hot(topk_indices, num_classes=self.num_experts).float()
    #     load = hard_assign.mean(dim=(0, 1, 2))
    #     target = 1.0 / self.num_experts
    #     return torch.mean((importance - target) ** 2) + torch.mean((load - target) ** 2)


    def _compute_aux_loss(self, router_logits, topk_indices):
        # 1. Get the routing probabilities (Importance)
        # Shape: [batch_size, sequence_length, num_experts]
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        # Average across the batch and sequence: [num_experts]
        P = router_probs.mean(dim=(0, 1))

        # 2. Get the actual assignments (Load)
        # Create a mask of which experts were chosen in top-k
        # topk_indices shape: [batch_size, sequence_length, k]
        # We want to know the fraction of tokens that went to each expert
        hard_assign = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        # Average across batch, sequence, and the 'k' dimension: [num_experts]
        f = hard_assign.mean(dim=(0, 1, 2))

        # 3. Compute the Loss
        # We multiply by num_experts so that the ideal loss is 1.0 (before the alpha multiplier)
        loss = self.num_experts * torch.sum(f * P)
        return loss

    def forward(self, hidden_states):
        router_logits = self.router(hidden_states)
        topk_values, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_values.float(), dim=-1).to(hidden_states.dtype)

        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_topk_indices = topk_indices.reshape(-1)
        flat_topk_weights = topk_weights.reshape(-1)
        token_positions = torch.arange(flat_hidden.size(0), device=hidden_states.device).repeat_interleave(
            self.top_k
        )

        sort_order = torch.argsort(flat_topk_indices)
        sorted_experts = flat_topk_indices.index_select(0, sort_order)
        sorted_token_positions = token_positions.index_select(0, sort_order)
        sorted_topk_weights = flat_topk_weights.index_select(0, sort_order)

        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        ends = counts.cumsum(0)
        starts = ends - counts

        mixed_flat = torch.zeros_like(flat_hidden)

        expert_ids = unique_experts.tolist()
        start_offsets = starts.tolist()
        end_offsets = ends.tolist()
        for expert_idx, s, e in zip(expert_ids, start_offsets, end_offsets):
            pos = sorted_token_positions[s:e]
            w = sorted_topk_weights[s:e].unsqueeze(-1)
            out = self.experts[expert_idx](flat_hidden.index_select(0, pos))
            mixed_flat.index_add_(0, pos, out * w)

        mixed_output = mixed_flat.view_as(hidden_states)
        self.last_aux_loss = self._compute_aux_loss(router_logits, topk_indices)
        # print(mixed_output)
        return mixed_output


class MoECausalLM(nn.Module):
    def __init__(
        self,
        model_name,
        moe_layer_indices,
        num_experts,
        top_k,
        router_aux_loss_weight,
        torch_dtype=torch.bfloat16,
        attn_implementation=None,
        base_model=None,
    ):
        super().__init__()
        if base_model is None:
            load_kwargs = {"torch_dtype": torch_dtype}
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.base_model = base_model
        self.router_aux_loss_weight = router_aux_loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_layers = []
        self.converted_layer_indices = []
        if moe_layer_indices:
            self._replace_mlp_with_moe(moe_layer_indices, num_experts, top_k)

    @property
    def config(self):
        return self.base_model.config

    @classmethod
    def _metadata_path(cls, model_name_or_path):
        local_dir = Path(str(model_name_or_path))
        if local_dir.is_dir():
            return local_dir / MOE_METADATA_FILENAME
        return None

    @classmethod
    def _load_saved_moe_metadata(cls, model_name_or_path):
        metadata_path = cls._metadata_path(model_name_or_path)
        if metadata_path is None or not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        moe_layer_indices=None,
        num_experts=None,
        top_k=None,
        router_aux_loss_weight=None,
        torch_dtype=torch.bfloat16,
        attn_implementation=None,
    ):
        saved_metadata = cls._load_saved_moe_metadata(model_name_or_path) or {}
        if moe_layer_indices is None:
            moe_layer_indices = saved_metadata.get("moe_layer_indices", [])
        if num_experts is None:
            num_experts = saved_metadata.get("num_experts", 4)
        if top_k is None:
            top_k = saved_metadata.get("top_k", 1)
        if router_aux_loss_weight is None:
            router_aux_loss_weight = saved_metadata.get("router_aux_loss_weight", 0.01)

        metadata_path = cls._metadata_path(model_name_or_path)
        has_local_moe_metadata = metadata_path is not None and metadata_path.exists()
        if has_local_moe_metadata and load_sharded_checkpoint is None:
            raise RuntimeError(
                "This checkpoint contains MoE metadata, but your Transformers version does not expose "
                "`load_sharded_checkpoint`. Upgrade Transformers to load local MoE checkpoints."
            )

        if not has_local_moe_metadata:
            return cls(
                model_name=model_name_or_path,
                moe_layer_indices=moe_layer_indices,
                num_experts=num_experts,
                top_k=top_k,
                router_aux_loss_weight=router_aux_loss_weight,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )

        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        config = AutoConfig.from_pretrained(str(metadata_path.parent))
        try:
            base_model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        except TypeError:
            base_model = AutoModelForCausalLM.from_config(config)
            if torch_dtype is not None:
                base_model = base_model.to(dtype=torch_dtype)
        model = cls(
            model_name=model_name_or_path,
            moe_layer_indices=moe_layer_indices,
            num_experts=num_experts,
            top_k=top_k,
            router_aux_loss_weight=router_aux_loss_weight,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            base_model=base_model,
        )
        load_sharded_checkpoint(model.base_model, str(metadata_path.parent), strict=True)
        return model

    def _get_transformer_blocks(self):
        # Qwen3 / LLaMA-style: model.model.layers
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            return self.base_model.model.layers
        # GPT-2-style: model.transformer.h
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            return self.base_model.transformer.h
        raise ValueError(
            "Unsupported model architecture. Expected either `model.model.layers` "
            "(Qwen3/LLaMA) or `model.transformer.h` (GPT-2)."
        )

    def _get_mlp(self, block):
        """Return the MLP sub-module from a transformer block, regardless of attribute name."""
        for attr in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, attr):
                return attr, getattr(block, attr)
        raise ValueError(f"Cannot find MLP in block: {type(block)}. Tried: mlp, feed_forward, ffn.")

    def _replace_mlp_with_moe(self, moe_layer_indices, num_experts, top_k):
        blocks = self._get_transformer_blocks()
        total_layers = len(blocks)

        for layer_idx in sorted(set(moe_layer_indices)):
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(f"Invalid layer index {layer_idx}. Valid range is [0, {total_layers - 1}].")

            mlp_attr, dense_mlp = self._get_mlp(blocks[layer_idx])
            hidden_size = self.base_model.config.hidden_size

            moe_mlp = MoEMLP(
                dense_mlp=dense_mlp,
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
            )
            setattr(blocks[layer_idx], mlp_attr, moe_mlp)
            self.moe_layers.append(moe_mlp)
            self.converted_layer_indices.append(layer_idx)

    def _get_router_aux_loss(self, device):
        if not self.moe_layers:
            return torch.tensor(0.0, device=device)
        aux_losses = [
            layer.last_aux_loss.to(device)
            for layer in self.moe_layers
            if layer.last_aux_loss is not None
        ]
        if not aux_losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(aux_losses).mean()

    def forward(self, *args, **kwargs):
        # print(kwargs)
        outputs = self.base_model(*args, **kwargs)
        if outputs.loss is not None and self.moe_layers and self.router_aux_loss_weight > 0.0:
            aux_loss = self._get_router_aux_loss(outputs.loss.device)
            outputs.loss = outputs.loss + self.router_aux_loss_weight * aux_loss
        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is not None:
            stripped_state_dict = {}
            for key, value in state_dict.items():
                normalized_key = key
                for prefix in ("module.", "base_model."):
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key[len(prefix):]
                stripped_state_dict[normalized_key] = value
            kwargs["state_dict"] = stripped_state_dict
        result = self.base_model.save_pretrained(save_directory, **kwargs)
        metadata = {
            "moe_layer_indices": self.converted_layer_indices,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "router_aux_loss_weight": self.router_aux_loss_weight,
        }
        metadata_path = Path(save_directory) / MOE_METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return result


def create_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If loading from a local MoE checkpoint, allow saved metadata to restore routing setup.
    explicit_moe = bool(args.moe_layer_indices)
    model = MoECausalLM.from_pretrained(
        model_name_or_path=args.model_name,
        moe_layer_indices=args.moe_layer_indices if explicit_moe else None,
        num_experts=args.num_experts if explicit_moe else None,
        top_k=args.moe_top_k if explicit_moe else None,
        router_aux_loss_weight=args.router_aux_loss_weight if explicit_moe else None,
        torch_dtype=getattr(args, "torch_dtype", torch.bfloat16),
        attn_implementation=getattr(args, "attn_implementation", None),
    )
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model.base_model.config, "use_cache"):
        model.base_model.config.use_cache = False
    # print(next(model.base_model.model.layers[26].mlp.router.parameters()).dtype)
    # print(next(model.base_model.model.layers[26].self_attn.q_proj.parameters()).dtype)
    return model, tokenizer
