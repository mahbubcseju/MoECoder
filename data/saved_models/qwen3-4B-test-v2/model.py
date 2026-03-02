import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Qwen3ForCausalLM, Qwen3Config, Qwen3Model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.modeling_utils import load_sharded_checkpoint
except ImportError:  # pragma: no cover - older transformers versions
    load_sharded_checkpoint = None


MOE_METADATA_FILENAME = "moe_config.json"


class MoEMLP(nn.Module):
    def __init__(self, dense_mlp, hidden_size, num_experts_temp, top_k):
        super().__init__()
        self.num_experts_temp = num_experts_temp
        self.top_k = top_k
        # inside MoEMLP.__init__
        ref = next(dense_mlp.parameters())
        self.router = nn.Linear(hidden_size, num_experts_temp, bias=False).to(
            device=ref.device, dtype=ref.dtype
        )
        self.experts = nn.ModuleList([copy.deepcopy(dense_mlp) for _ in range(num_experts_temp)])
        self.last_aux_loss = None
        self.last_statement_loss = None

    # def _compute_aux_loss(self, router_logits, topk_indices):
    #     router_probs = torch.softmax(router_logits.float(), dim=-1)
    #     importance = router_probs.mean(dim=(0, 1))
    #     hard_assign = F.one_hot(topk_indices, num_classes=self.num_experts_temp).float()
    #     load = hard_assign.mean(dim=(0, 1, 2))
    #     target = 1.0 / self.num_experts_temp
    #     return torch.mean((importance - target) ** 2) + torch.mean((load - target) ** 2)

    def _compute_statement_loss(self, router_logits, temperature=0.07, eps=1e-8):
        """
        router_logits: [B, T, E]
        concept_mat:   [B, T, 12] or [N, 12]
        returns: scalar loss
        """
        concept_mat = getattr(self, "concept_mat", None)
        if concept_mat is None:
            # print("concept_mat not found in MoEMLP")
            return router_logits.new_tensor(0.0)
    
        B, T, E = router_logits.shape
        N = B * T
        
        # print("57", concept_mat.shape, router_logits.shape)
        # Flatten tokens
        # probs = torch.softmax(router_logits.float(), dim=-1).reshape(N, E)  # [N, E]
        router_logits = router_logits.float().reshape(N, E)
        probs = F.normalize(router_logits, p=2, dim=-1)  # cosine similarity in expert-prob space

        if concept_mat.dim() == 3:
            C = concept_mat.reshape(N, concept_mat.size(-1))
        else:
            C = concept_mat
        C = C.to(router_logits.device)

        # valid tokens: not all -1
        valid = ~(C.eq(-1).all(dim=-1))  # [N]
        # print("valid tokens:", valid.sum().item())
        if valid.sum() < 2:
            return router_logits.new_tensor(0.0)

        # Keep only valid tokens
        probs_v = probs[valid]           # [Nv, E]
        C_v = C[valid]                   # [Nv, 12]

        # print("79", probs_v.shape, C_v.shape)
        Nv = probs_v.size(0)

        # Multi-hot mask of active concepts
        C01 = C_v.eq(1)                  # [Nv, 12]

        # Positive pairs: share at least one concept==1
        # shared = C01 @ C01.T gives count of shared concepts
        shared = (C01.float() @ C01.float().T)  # [Nv, Nv]
        pos = shared.gt(0)                      # [Nv, Nv]

        # Remove self-pairs
        eye = torch.eye(Nv, dtype=torch.bool, device=router_logits.device)
        pos = pos & ~eye

        # If a token has no positives, skip it (avoid degenerate anchors)
        has_pos = pos.any(dim=1)
        if has_pos.sum() == 0:
            return router_logits.new_tensor(0.0)

        probs_a = probs_v[has_pos]          # anchors
        pos_a = pos[has_pos]                # [Na, Nv]

        # print("102", probs_a.shape, pos_a.shape)

        # Similarity matrix [Na, Nv]
        sim = (probs_a @ probs_v.T) / temperature

        # Log-softmax over all candidates for each anchor
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [Na, Nv]

        # Average log-prob over positives
        # For each anchor i: loss_i = - mean_{j in Pos(i)} logp[i,j]
        pos_counts = pos_a.sum(dim=1).clamp_min(1)
        loss = -(logp.masked_fill(~pos_a, 0.0).sum(dim=1) / pos_counts)

        return loss.mean()



    def _compute_aux_loss(self, router_logits, topk_indices):
        # 1. Get the routing probabilities (Importance)
        # Shape: [batch_size, sequence_length, num_experts_temp]
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        # Average across the batch and sequence: [num_experts_temp]
        P = router_probs.mean(dim=(0, 1))

        # 2. Get the actual assignments (Load)
        # Create a mask of which experts were chosen in top-k
        # topk_indices shape: [batch_size, sequence_length, k]
        # We want to know the fraction of tokens that went to each expert
        hard_assign = F.one_hot(topk_indices, num_classes=self.num_experts_temp).float()
        # Average across batch, sequence, and the 'k' dimension: [num_experts_temp]
        f = hard_assign.mean(dim=(0, 1, 2))

        # 3. Compute the Loss
        # We multiply by num_experts_temp so that the ideal loss is 1.0 (before the alpha multiplier)
        loss = self.num_experts_temp * torch.sum(f * P)
        return loss

    def forward(self, hidden_states):
        """
        Compute outputs for ALL experts for ALL tokens, but merge using only top-k experts per token.
        Shapes:
        hidden_states: [B, T, H]
        router_logits: [B, T, E]
        expert_outputs: [B, T, E, H]
        final: [B, T, H]
        """
        B, T, H = hidden_states.shape
        E = self.num_experts_temp
        K = self.top_k

        # 1) Router
        router_logits = self.router(hidden_states)  # [B, T, E]

        # 2) Top-k selection + weights
        topk_values, topk_indices = torch.topk(router_logits, k=K, dim=-1)  # [B, T, K]
        topk_weights = torch.softmax(topk_values.float(), dim=-1).to(hidden_states.dtype)  # [B, T, K]

        # 3) Run ALL experts on ALL tokens
        # Each expert returns [B, T, H]
        all_expert_outs = []
        for e in range(E):
            all_expert_outs.append(self.experts[e](hidden_states))
            # print()
        # Stack -> [B, T, E, H]
        expert_outputs = torch.stack(all_expert_outs, dim=2)

        # 4) Gather the top-k expert outputs for each token
        # topk_indices: [B, T, K] -> expand to [B, T, K, H] for gather
        gather_idx = topk_indices.unsqueeze(-1).expand(B, T, K, H)  # [B, T, K, H]
        topk_expert_outputs = expert_outputs.gather(dim=2, index=gather_idx)  # [B, T, K, H]

        # 5) Weighted merge of only top-k
        mixed_output = (topk_expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, H]

        self.last_aux_loss = self._compute_aux_loss(router_logits, topk_indices)
        self.last_statement_loss = self._compute_statement_loss(router_logits)
        # print
        # assert mixed_output== torch.rand_like(mixed_output)
        # return mixed_output == torch.rand_like(mixed_output)
        # return torch.rand_like(mixed_output)
        return mixed_output


class Qwen3MoEConfig(Qwen3Config):
    """
    Qwen3Config + extra fields for your MoE routing.
    """
    # Important choice:
    # - If you set a NEW model_type (e.g. "qwen3_moe"), you must register it with AutoConfig.
    # - But vLLM typically keys off model_type to pick a built-in architecture. If vLLM doesn't
    #   know "qwen3_moe", it will fail unless you add a vLLM custom model.
    model_type = "qwen3"

    def __init__(
        self,
        moe_layer_indices=None,
        num_experts_temp=4,
        top_k=1,
        router_aux_loss_weight=0.01,
        add_expert_mlp=False,  # signal to model init to build MoE right away (vs. convert later)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moe_layer_indices = moe_layer_indices or []
        self.num_experts_temp = int(num_experts_temp)
        self.top_k = int(top_k)
        self.router_aux_loss_weight = float(router_aux_loss_weight)
        self.add_expert_mlp = add_expert_mlp
        


class Qwen3MoEModel(Qwen3Model):
    config_class = Qwen3MoEConfig

    def __init__(self, config):
        super().__init__(config)

        moe_layer_indices = getattr(config, "moe_layer_indices", [])
        num_experts = getattr(config, "num_experts_temp", 4)
        top_k = getattr(config, "top_k", 1)
        self.moe_layers  = []
        self.converted_layer_indices = []

        if moe_layer_indices and getattr(config, "add_expert_mlp", False) == True:
            hidden_size = config.hidden_size
            for layer_idx in moe_layer_indices:
                dense_mlp = self.layers[layer_idx].mlp
                self.layers[layer_idx].mlp = MoEMLP(
                    dense_mlp=dense_mlp,
                    hidden_size=hidden_size,
                    num_experts_temp=num_experts,
                    top_k=top_k,
                )
                self.moe_layers.append(self.layers[layer_idx].mlp)
                self.converted_layer_indices.append(layer_idx)
        print("initialized MoE model with config:", config)


class MoECausalLM(Qwen3ForCausalLM):
    config_class = Qwen3MoEConfig
    def __init__(
        self,
        config
    ):  
        self._config = config  # store config early to avoid issues in super().__init__
        super().__init__(config)

        self.model = Qwen3MoEModel(config)
        self.moe_layers = self.model.moe_layers or []
        self.converted_layer_indices = self.model.converted_layer_indices or []
        # if you want tying like HF does:
        self.lm_head.weight = self.model.embed_tokens.weight
        print(self.model)

    @property
    def config(self):
        if hasattr(self, "model") and self.model is not None:
            return self.model.config
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        if hasattr(self, "model") and self.model is not None:
            self.model.config = value

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
    def from_pretrained(cls, path, *, moe_layer_indices=None, num_experts_temp=None,
                        top_k=None, router_aux_loss_weight=None, **kwargs):

        config = AutoConfig.from_pretrained(path, **kwargs)

        # user overrides (explicit)
        if moe_layer_indices is not None: config.moe_layer_indices = moe_layer_indices
        if num_experts_temp is not None: config.num_experts_temp = num_experts_temp
        if top_k is not None: config.top_k = top_k
        if router_aux_loss_weight is not None: config.router_aux_loss_weight = router_aux_loss_weight

        # check if this directory contains a MoE checkpoint
        meta_path = Path(str(path)) / MOE_METADATA_FILENAME
        saved_meta = None
        if meta_path.exists():
            saved_meta = json.loads(meta_path.read_text())

        # If we are loading a saved MoE checkpoint (and user didn't explicitly override to "dense"):
        loading_moe = saved_meta is not None and moe_layer_indices is None

        if loading_moe:
            # set config from saved metadata
            config.moe_layer_indices = saved_meta["moe_layer_indices"]
            config.num_experts_temp = saved_meta["num_experts_temp"]
            config.top_k = saved_meta["top_k"]
            config.router_aux_loss_weight = saved_meta.get("router_aux_loss_weight", 0.0)

            # 1) build model WITH MoE in __init__
            model = cls(config)

            # 2) now load weights into existing MoE modules
            state_dict = kwargs.pop("state_dict", None)
            if state_dict is None:
                # let HF fetch weights then load; we can reuse the internal loader:
                model = super(Qwen3ForCausalLM, model).from_pretrained(path, config=config, **kwargs)
                # NOTE: this line is illustrative; see note below.
            else:
                model.load_state_dict(state_dict, strict=True)

            return model

        else:

            print("**************************************************")
            # dense checkpoint load first
            config.add_expert_mlp = False  # signal to __init__ to build MoE right away
            model = super().from_pretrained(path, config=config, **kwargs)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            # then convert if requested
            if getattr(config, "moe_layer_indices", []):
                model._replace_mlp_with_moe(config.moe_layer_indices, config.num_experts_temp, config.top_k)
                model.router_aux_loss_weight = config.router_aux_loss_weight
                model.top_k = config.top_k
                model.num_experts_temp = config.num_experts_temp

                # initialize experts from the (already-loaded) dense mlp
                # (your copy logic here)

            return model
    

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path,
    #     **kwargs,
    # ):
    #     # 1) Load config (or use provided)
    #     if config is None:
    #         config = Qwen3MoEConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    #     # # 2) Pull saved MoE metadata (optional extra file)
    #     # saved = cls._load_saved_moe_metadata(pretrained_model_name_or_path) or {}

    #     # 3) Resolve MoE params: explicit arg > config > saved metadata > default
    #     def pick(name, explicit, default):
    #         if explicit is not None:
    #             return explicit
    #         if hasattr(config, name):
    #             return getattr(config, name)
    #         # if name in saved:
    #         #     return saved[name]
    #         return default

    #     config.moe_layer_indices = pick("moe_layer_indices", moe_layer_indices, [])
    #     config.num_experts_temp = pick("num_experts_temp", num_experts_temp, 4)
    #     config.top_k = pick("top_k", top_k, 1)
    #     config.router_aux_loss_weight = pick("router_aux_loss_weight", router_aux_loss_weight, 0.01)

    #     # 4) Pass standard HF loading args
    #     if torch_dtype is not None:
    #         kwargs["torch_dtype"] = torch_dtype
    #     if attn_implementation is not None:
    #         kwargs["attn_implementation"] = attn_implementation

    #     # 5) Let HF build + load weights
    #     return super().from_pretrained(
    #         pretrained_model_name_or_path,
    #         config=config,
    #         **kwargs,
    #     )
    

    def _get_transformer_blocks(self):
        # For Qwen3, transformer blocks are on self.model.layers
        if hasattr(self, "model") and hasattr(self.model, "layers"):
            return self.model.layers
        raise ValueError("Unsupported Qwen3 architecture: expected self.model.layers")


    def _get_mlp(self, block):
        """Return the MLP sub-module from a transformer block, regardless of attribute name."""
        for attr in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, attr):
                return attr, getattr(block, attr)
        raise ValueError(f"Cannot find MLP in block: {type(block)}. Tried: mlp, feed_forward, ffn.")

    def _replace_mlp_with_moe(self, moe_layer_indices, num_experts_temp, top_k):
        blocks = self._get_transformer_blocks()
        total_layers = len(blocks)

        for layer_idx in sorted(set(moe_layer_indices)):
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(f"Invalid layer index {layer_idx}. Valid range is [0, {total_layers - 1}].")

            mlp_attr, dense_mlp = self._get_mlp(blocks[layer_idx])
            hidden_size = self.model.config.hidden_size

            moe_mlp = MoEMLP(
                dense_mlp=dense_mlp,
                hidden_size=hidden_size,
                num_experts_temp=num_experts_temp,
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

    def _get_router_statement_loss(self, device):
        if not self.moe_layers:
            return torch.tensor(0.0, device=device)
        aux_losses = [
            layer.last_statement_loss.to(device)
            for layer in self.moe_layers
            if layer.last_statement_loss is not None
        ]
        if not aux_losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(aux_losses).mean()

    def forward(self, *args, **kwargs):
        # print(kwargs)
        # print(kwargs.get("concept_mat", None))
        for layer in self.moe_layers:
            setattr(layer, "concept_mat", kwargs.get("concept_mat", None))
            # layer.set
        outputs = super().forward(*args, **kwargs)
        if outputs.loss is not None and self.moe_layers and self.router_aux_loss_weight > 0.0:
            aux_loss = self._get_router_aux_loss(outputs.loss.device)
            statement_loss = self._get_router_statement_loss(outputs.loss.device)
            # print(statement_loss)
            outputs.loss = outputs.loss + self.router_aux_loss_weight * aux_loss + self.router_aux_loss_weight * 2 * statement_loss
        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is not None:
            stripped_state_dict = {}
            for key, value in state_dict.items():
                normalized_key = key
                print(key)
                # for prefix in ("module.", "base_model."):
                #     if normalized_key.startswith(prefix):
                #         normalized_key = normalized_key[len(prefix):]
                stripped_state_dict[normalized_key] = value
            kwargs["state_dict"] = stripped_state_dict
        result = self.model.save_pretrained(save_directory, **kwargs)
        metadata = {
            "moe_layer_indices": self.converted_layer_indices,
            "num_experts_tem[p": self.num_experts_temp,
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
        args.model_name,
        moe_layer_indices=args.moe_layer_indices if explicit_moe else None,
        num_experts_temp=args.num_experts_temp if explicit_moe else None,
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
    print(model)
    # print(next(model.base_model.model.layers[26].mlp.router.parameters()).dtype)
    # print(next(model.base_model.model.layers[26].self_attn.q_proj.parameters()).dtype)
    return model, tokenizer
