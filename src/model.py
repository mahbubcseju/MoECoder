import copy
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

try:
    from .modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config, Qwen3Model
except ImportError:
    from modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config, Qwen3Model


MOE_METADATA_FILENAME = "moe_config.json"


class MoEMLP(nn.Module):
    """
    Hybrid MoE MLP with two independent expert pools:

    Code experts  (num_concepts + 1):
      - Experts 0 … num_concepts-1 each own one concept type.
      - Expert num_concepts is the catch-all for code tokens not covered by any concept.
      - Routing is deterministic: driven by concept_mat + code_mask, no router network.
      - A token belonging to k active concepts is sent to all k experts; outputs are averaged.

    NL experts  (num_nl_experts):
      - Receive every token that is NOT a code token.
      - Routing is learned (nl_router) with top-k selection.
      - Load-balancing aux loss is applied to this pool only.

    When concept_mat / code_mask are not set (e.g. DPO, eval), all tokens fall through
    to the NL expert pool.
    """

    def __init__(self, dense_mlp, hidden_size, num_concepts, num_nl_experts, top_k_nl=1, top_k_code=None):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_code_experts = num_concepts + 1  # +1 for "other" code tokens
        self.num_nl_experts = num_nl_experts
        self.top_k_nl = top_k_nl
        self.top_k_code = num_concepts if top_k_code is None else top_k_code

        self.code_experts = nn.ModuleList([
            copy.deepcopy(dense_mlp) for _ in range(self.num_code_experts)
        ])
        self.nl_experts = nn.ModuleList([
            copy.deepcopy(dense_mlp) for _ in range(num_nl_experts)
        ])
        ref = next(dense_mlp.parameters())
        self.nl_router = nn.Linear(hidden_size, num_nl_experts, bias=False).to(
            device=ref.device, dtype=ref.dtype
        )
        self.last_aux_loss = None

    def _compute_nl_aux_loss(self, router_logits, topk_indices, nl_mask=None):
        """Load-balancing loss computed only over NL (non-code) tokens."""
        B, T, E = router_logits.shape
        if nl_mask is not None:
            flat_mask  = nl_mask.reshape(-1)                                 # [B*T]
            flat_logits = router_logits.reshape(B * T, E)[flat_mask]         # [N_nl, E]
            flat_topk   = topk_indices.reshape(B * T, -1)[flat_mask]         # [N_nl, k]
        else:
            flat_logits = router_logits.reshape(B * T, E)
            flat_topk   = topk_indices.reshape(B * T, -1)

        if flat_logits.shape[0] == 0:
            return router_logits.new_tensor(0.0)

        router_probs = torch.softmax(flat_logits.float(), dim=-1)
        P = router_probs.mean(dim=0)                                          # [E]
        hard_assign = F.one_hot(flat_topk, num_classes=self.num_nl_experts).float()
        f = hard_assign.mean(dim=(0, 1))                                      # [E]
        return self.num_nl_experts * torch.sum(f * P)

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        code_mask   = getattr(self, "code_mask",   None)  # [B, T] bool
        concept_mat = getattr(self, "concept_mat", None)  # [B, T, C] long

        # --- NL experts (always computed; masked out for code tokens when routing is active) ---
        nl_logits = self.nl_router(hidden_states)                             # [B, T, E_nl]
        topk_vals, topk_idx = torch.topk(nl_logits, k=self.top_k_nl, dim=-1)
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(hidden_states.dtype)
        nl_expert_outs = torch.stack(
            [self.nl_experts[e](hidden_states)[0] for e in range(self.num_nl_experts)], dim=2
        )                                                                      # [B, T, E_nl, H]
        gather_idx = topk_idx.unsqueeze(-1).expand(B, T, self.top_k_nl, H)
        topk_nl_outs = nl_expert_outs.gather(dim=2, index=gather_idx)         # [B, T, k, H]
        nl_output = (topk_nl_outs * topk_w.unsqueeze(-1)).sum(dim=2)          # [B, T, H]

        if concept_mat is not None and code_mask is not None:
            # --- Code experts (concept-driven deterministic routing) ---
            # print("Concept mask", hidden_states.shape)
            cm      = concept_mat.to(hidden_states.device)   # [B, T, C]
            cm_code = code_mask.to(hidden_states.device)     # [B, T] bool

            code_expert_outs = torch.stack(
                [self.code_experts[e](hidden_states)[0] for e in range(self.num_code_experts)], dim=2
            )                                                                  # [B, T, C+1, H]

            # Per-token weights for each concept expert.
            # cm > 0 means active; the value is the minimum overlapping span length
            # (smaller = more specific / closer concept).
            is_active = (cm > 0)                                              # [B, T, C] bool

            if self.top_k_code < self.num_concepts:
                # Pick the top_k_code concepts with the SMALLEST span length per token.
                # Non-active entries are masked to +inf so they are never selected.
                span_scores = cm.float().masked_fill(~is_active, float('inf'))
                _, topk_code_idx = torch.topk(span_scores, k=self.top_k_code, dim=-1, largest=False)
                selected = torch.zeros(B, T, self.num_concepts, dtype=torch.bool, device=cm.device)
                selected.scatter_(-1, topk_code_idx, True)
                # Only keep concepts that are both selected AND truly active
                is_active = is_active & selected

            active     = is_active.float()                                    # [B, T, C]
            num_active = active.sum(dim=-1, keepdim=True).clamp(min=1.0)      # [B, T, 1]
            conc_w     = active / num_active                                  # [B, T, C]

            # "Other" code expert: code token with no active concept
            has_concept = is_active.any(dim=-1)                               # [B, T]
            other_w     = (cm_code & ~has_concept).float().unsqueeze(-1)      # [B, T, 1]

            # Combined routing weights [B, T, C+1]; zero for NL tokens
            code_w = torch.cat([conc_w, other_w], dim=-1)                     # [B, T, C+1]
            code_w = (code_w * cm_code.float().unsqueeze(-1)).to(hidden_states.dtype)

            code_output = (code_expert_outs * code_w.unsqueeze(-1)).sum(dim=2)  # [B, T, H]

            # NL output only for non-code tokens
            nl_output = nl_output * (~cm_code).float().unsqueeze(-1).to(hidden_states.dtype)

            output = code_output + nl_output
            self.last_aux_loss = self._compute_nl_aux_loss(nl_logits, topk_idx, ~cm_code)
        else:
            # print("No concept mask available", hidden_states.shape)
            # Fallback: no routing info available (DPO, inference) → all tokens use NL experts
            output = nl_output
            self.last_aux_loss = self._compute_nl_aux_loss(nl_logits, topk_idx, None)

        return output, self.last_aux_loss, nl_logits.new_tensor(0.0)


class Qwen3MoEConfig(Qwen3Config):
    model_type = "qwen3"

    def __init__(
        self,
        moe_layer_indices=None,
        num_concepts=14,
        num_nl_experts=4,
        top_k_nl=1,
        top_k_code=None,
        router_aux_loss_weight=0.01,
        add_expert_mlp=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moe_layer_indices      = moe_layer_indices or []
        self.num_concepts           = int(num_concepts)
        self.num_nl_experts         = int(num_nl_experts)
        self.top_k_nl               = int(top_k_nl)
        self.top_k_code             = int(num_concepts if top_k_code is None else top_k_code)
        self.router_aux_loss_weight = float(router_aux_loss_weight)
        self.add_expert_mlp         = add_expert_mlp


class Qwen3MoEModel(Qwen3Model):
    config_class = Qwen3MoEConfig

    def __init__(self, config):
        super().__init__(config)

        moe_layer_indices = getattr(config, "moe_layer_indices", [])
        self.moe_layers              = []
        self.converted_layer_indices = []
        self.router_aux_loss_weight  = 0
        self.num_nl_experts          = 0
        self.num_concepts            = 0

        if moe_layer_indices and getattr(config, "add_expert_mlp", False):
            num_concepts    = getattr(config, "num_concepts",   14)
            num_nl_experts  = getattr(config, "num_nl_experts", 4)
            top_k_nl        = getattr(config, "top_k_nl",       1)
            top_k_code      = getattr(config, "top_k_code",     num_concepts)
            hidden_size     = config.hidden_size
            for layer_idx in moe_layer_indices:
                dense_mlp = self.layers[layer_idx].mlp
                self.layers[layer_idx].mlp = MoEMLP(
                    dense_mlp      = dense_mlp,
                    hidden_size    = hidden_size,
                    num_concepts   = num_concepts,
                    num_nl_experts = num_nl_experts,
                    top_k_nl       = top_k_nl,
                    top_k_code     = top_k_code,
                )
                self.moe_layers.append(self.layers[layer_idx].mlp)
                self.converted_layer_indices.append(layer_idx)
        print("initialized MoE model with config:", config)


class MoECausalLM(Qwen3ForCausalLM):
    config_class = Qwen3MoEConfig

    def __init__(self, config):
        self._config = config
        super().__init__(config)

        self.model = Qwen3MoEModel(config)
        self.moe_layers              = self.model.moe_layers
        self.converted_layer_indices = self.model.converted_layer_indices
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
    def _load_full_state_dict(cls, checkpoint_dir, torch_dtype=None):
        """Read every tensor from all safetensors shards into one state dict."""
        from safetensors import safe_open
        ckpt_dir = Path(str(checkpoint_dir))
        sf_files = sorted(ckpt_dir.glob("*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {ckpt_dir}")
        state_dict: dict[str, torch.Tensor] = {}
        for sf in sf_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    state_dict[key] = t.to(torch_dtype) if torch_dtype is not None else t
        return state_dict

    @classmethod
    def _load_moe_weights(cls, model, checkpoint_dir, torch_dtype=None):
        """Load only expert and router weights from a checkpoint into an already-converted MoE model."""
        from safetensors import safe_open

        ckpt_dir = Path(str(checkpoint_dir))
        sf_files = sorted(ckpt_dir.glob("*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {ckpt_dir}")

        moe_state: dict[str, torch.Tensor] = {}
        for sf in sf_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if any(s in key for s in (".code_experts.", ".nl_experts.", ".nl_router.")):
                        t = f.get_tensor(key)
                        moe_state[key] = t.to(torch_dtype) if torch_dtype is not None else t

        if not moe_state:
            return

        missing, unexpected = model.load_state_dict(moe_state, strict=False)
        if unexpected:
            print(f"[_load_moe_weights] unexpected keys (first 5): {unexpected[:5]}")

    @classmethod
    def from_pretrained(cls, path, *, moe_layer_indices=None, num_concepts=None,
                        num_nl_experts=None, top_k_nl=None, top_k_code=None,
                        router_aux_loss_weight=None, **kwargs):

        config = AutoConfig.from_pretrained(path, **kwargs)

        if moe_layer_indices      is not None: config.moe_layer_indices      = moe_layer_indices
        if num_concepts           is not None: config.num_concepts           = num_concepts
        if num_nl_experts         is not None: config.num_nl_experts         = num_nl_experts
        if top_k_nl               is not None: config.top_k_nl               = top_k_nl
        if top_k_code             is not None: config.top_k_code             = top_k_code
        if router_aux_loss_weight is not None: config.router_aux_loss_weight = router_aux_loss_weight

        meta_path  = Path(str(path)) / MOE_METADATA_FILENAME
        saved_meta = None
        if meta_path.exists():
            saved_meta = json.loads(meta_path.read_text())

        loading_moe = saved_meta is not None and moe_layer_indices is None

        if loading_moe:
            config.moe_layer_indices      = saved_meta["moe_layer_indices"]
            config.num_concepts           = saved_meta.get("num_concepts",           14)
            config.num_nl_experts         = saved_meta.get("num_nl_experts",         4)
            config.top_k_nl               = saved_meta.get("top_k_nl",               1)
            config.top_k_code             = saved_meta.get("top_k_code",             config.num_concepts)
            config.router_aux_loss_weight = saved_meta.get("router_aux_loss_weight", 0.0)
            config.add_expert_mlp         = True

            torch_dtype = kwargs.get("torch_dtype", None)
            model = cls(config)
            if torch_dtype is not None:
                model = model.to(torch_dtype)

            state_dict = kwargs.pop("state_dict", None)
            if state_dict is None:
                state_dict = cls._load_full_state_dict(path, torch_dtype=torch_dtype)
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if missing:
                print(f"[from_pretrained] missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"[from_pretrained] unexpected keys (first 5): {unexpected[:5]}")

            model.router_aux_loss_weight = config.router_aux_loss_weight
            model.top_k_nl               = config.top_k_nl
            model.top_k_code             = config.top_k_code
            model.num_nl_experts         = config.num_nl_experts
            model.num_concepts           = config.num_concepts
            return model

        else:
            print("**************************************************")
            config.add_expert_mlp = False
            model = super().from_pretrained(path, config=config, **kwargs)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            if getattr(config, "moe_layer_indices", []):
                num_concepts   = getattr(config, "num_concepts",           14)
                num_nl_experts = getattr(config, "num_nl_experts",          4)
                top_k_nl       = getattr(config, "top_k_nl",                1)
                top_k_code     = getattr(config, "top_k_code",    num_concepts)
                aux_weight     = getattr(config, "router_aux_loss_weight", 0.01)
                model._replace_mlp_with_moe(
                    config.moe_layer_indices,
                    num_concepts,
                    num_nl_experts,
                    top_k_nl,
                    top_k_code,
                )
                model.router_aux_loss_weight = aux_weight
                model.top_k_nl               = top_k_nl
                model.top_k_code             = top_k_code
                model.num_nl_experts         = num_nl_experts
                model.num_concepts           = num_concepts

                if saved_meta is not None:
                    torch_dtype = kwargs.get("torch_dtype", None)
                    cls._load_moe_weights(model, path, torch_dtype=torch_dtype)

            return model

    def _get_transformer_blocks(self):
        if hasattr(self, "model") and hasattr(self.model, "layers"):
            return self.model.layers
        raise ValueError("Unsupported architecture: expected self.model.layers")

    def _get_mlp(self, block):
        for attr in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, attr):
                return attr, getattr(block, attr)
        raise ValueError(f"Cannot find MLP in block: {type(block)}")

    def _replace_mlp_with_moe(self, moe_layer_indices, num_concepts, num_nl_experts, top_k_nl, top_k_code=None):
        blocks = self._get_transformer_blocks()
        total_layers = len(blocks)
        for layer_idx in sorted(set(moe_layer_indices)):
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(f"Invalid layer index {layer_idx}. Valid range: [0, {total_layers - 1}].")
            mlp_attr, dense_mlp = self._get_mlp(blocks[layer_idx])
            moe_mlp = MoEMLP(
                dense_mlp      = dense_mlp,
                hidden_size    = self.model.config.hidden_size,
                num_concepts   = num_concepts,
                num_nl_experts = num_nl_experts,
                top_k_nl       = top_k_nl,
                top_k_code     = top_k_code,
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

    def forward(self, *args, loss_weights=None, concept_mat=None, code_mask=None,
                skip_moe_losses=False, **kwargs):
        # During decode steps (KV cache active, processing 1 new generated token),
        # concept routing is not applicable — generated tokens have no annotations.
        # Only use concept routing during prefill (processing the full prompt).
        past_kv   = kwargs.get("past_key_values")
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        is_decode_step = (past_kv is not None
                          and input_ids is not None
                          and input_ids.shape[1] == 1)
        use_concept = not skip_moe_losses and not is_decode_step
        # print(use_concept, concept_mat)
        for layer in self.moe_layers:
            setattr(layer, "concept_mat", concept_mat if use_concept else None)
            setattr(layer, "code_mask",   code_mask   if use_concept else None)

        outputs = super().forward(*args, **kwargs)

        # Weighted cross-entropy: differentiates reasoning vs content tokens
        if outputs.loss is not None and loss_weights is not None:
            labels        = kwargs.get("labels")
            shift_logits  = outputs.logits[..., :-1, :].contiguous()
            shift_labels  = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            weighted = per_token_loss * shift_weights * mask
            outputs.loss = weighted.sum() / (shift_weights * mask).sum().clamp(min=1e-8)

        # NL load-balancing aux loss
        if (not skip_moe_losses
                and outputs.loss is not None
                and self.moe_layers
                and self.router_aux_loss_weight > 0.0):
            aux_loss = self._get_router_aux_loss(outputs.logits.device)
            outputs.loss = outputs.loss + self.router_aux_loss_weight * aux_loss

        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is not None:
            kwargs["state_dict"] = {key: value for key, value in state_dict.items()}

        self.config.architectures = ["MoECausalLM"]
        self.config.auto_map = {
            "AutoConfig":          "model.Qwen3MoEConfig",
            "AutoModel":           "model.Qwen3MoEModel",
            "AutoModelForCausalLM": "model.MoECausalLM",
        }
        self.config.add_expert_mlp = bool(self.moe_layers)
        self.config.use_cache = True

        result = super().save_pretrained(save_directory, **kwargs)
        num_concepts = getattr(self, "num_concepts", 14)
        metadata = {
            "moe_layer_indices":      getattr(self, "converted_layer_indices", []),
            "num_concepts":           num_concepts,
            "num_nl_experts":         getattr(self, "num_nl_experts",          4),
            "top_k_nl":               getattr(self, "top_k_nl",                1),
            "top_k_code":             getattr(self, "top_k_code",              num_concepts),
            "router_aux_loss_weight": getattr(self, "router_aux_loss_weight",  0.01),
        }
        metadata_path = Path(save_directory) / MOE_METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        src_dir = Path(__file__).resolve().parent
        for fname in ("model.py", "modeling_qwen3.py"):
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, Path(save_directory) / fname)

        return result


def create_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    explicit_moe = bool(args.moe_layer_indices)
    model = MoECausalLM.from_pretrained(
        args.model_name,
        moe_layer_indices      = args.moe_layer_indices      if explicit_moe else None,
        num_concepts           = args.num_concepts            if explicit_moe else None,
        num_nl_experts         = args.num_nl_experts          if explicit_moe else None,
        top_k_nl               = args.moe_top_k              if explicit_moe else None,
        top_k_code             = args.top_k_code             if explicit_moe else None,
        router_aux_loss_weight = args.router_aux_loss_weight if explicit_moe else None,
        torch_dtype         = getattr(args, "torch_dtype", torch.bfloat16),
        attn_implementation = getattr(args, "attn_implementation", None),
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing with use_reentrant=False for better vLLM compatibility.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer
