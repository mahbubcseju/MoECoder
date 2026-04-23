import copy
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config, Qwen3Model
from transformers import AutoConfig, AutoTokenizer


MOE_METADATA_FILENAME = "moe_config.json"

# ---------------------------------------------------------------------------
# Shared loss helpers (used by both MoEMLP and MoELoRALinear)
# ---------------------------------------------------------------------------

def _moe_aux_loss(router_logits: torch.Tensor, topk_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Load-balancing auxiliary loss.  Encourages uniform expert utilisation.

    router_logits : [B, T, E]
    topk_indices  : [B, T, K]
    """
    router_probs = torch.softmax(router_logits.float(), dim=-1)
    P = router_probs.mean(dim=(0, 1))                                    # [E]
    hard_assign = F.one_hot(topk_indices, num_classes=num_experts).float()
    f = hard_assign.mean(dim=(0, 1, 2))                                  # [E]
    return num_experts * torch.sum(f * P)


def _moe_statement_loss(
    router_logits: torch.Tensor,
    concept_mat,                    # [B, T, C] or None
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss: tokens sharing a code concept should
    receive similar router distributions.

    router_logits : [B, T, E]
    concept_mat   : [B, T, C]  values in {-1, 0, 1}
    """
    if concept_mat is None:
        return router_logits.new_tensor(0.0)

    B, T, E = router_logits.shape
    N = B * T

    rl = router_logits.float().reshape(N, E)
    probs = F.normalize(rl, p=2, dim=-1)   # cosine in expert-prob space

    C = concept_mat.reshape(N, -1).to(router_logits.device)

    valid = ~(C.eq(-1).all(dim=-1))        # tokens covered by ≥1 concept
    if valid.sum() < 2:
        return router_logits.new_tensor(0.0)

    probs_v = probs[valid]                 # [Nv, E]
    C_v = C[valid]                         # [Nv, num_concepts]
    Nv = probs_v.size(0)

    C01 = C_v.eq(1)                        # multi-hot active concepts
    shared = C01.float() @ C01.float().T   # [Nv, Nv]  — shared concept count
    pos = shared.gt(0)
    eye = torch.eye(Nv, dtype=torch.bool, device=rl.device)
    pos = pos & ~eye                       # exclude self-pairs

    has_pos = pos.any(dim=1)
    if has_pos.sum() == 0:
        return router_logits.new_tensor(0.0)

    probs_a = probs_v[has_pos]             # anchor embeddings
    pos_a = pos[has_pos]

    sim = (probs_a @ probs_v.T) / temperature
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos_a.sum(dim=1).clamp_min(1)
    loss = -(logp.masked_fill(~pos_a, 0.0).sum(dim=1) / pos_counts)
    return loss.mean()


# ---------------------------------------------------------------------------
# MoEMLP  –  replaces an entire MLP block with a mixture of MLP experts
# ---------------------------------------------------------------------------

class MoEMLP(nn.Module):
    def __init__(self, dense_mlp, hidden_size, num_experts_temp, top_k):
        super().__init__()
        self.num_experts_temp = num_experts_temp
        self.top_k = top_k
        ref = next(dense_mlp.parameters())
        self.router = nn.Linear(hidden_size, num_experts_temp, bias=False).to(
            device=ref.device, dtype=ref.dtype
        )
        self.experts = nn.ModuleList([copy.deepcopy(dense_mlp) for _ in range(num_experts_temp)])
        self.last_aux_loss = None
        self.last_statement_loss = None

    def forward(self, hidden_states):
        """
        hidden_states : [B, T, H]
        returns       : mixed_output [B, T, H], aux_loss scalar, statement_loss scalar
        """
        B, T, H = hidden_states.shape
        E = self.num_experts_temp
        K = self.top_k

        router_logits = self.router(hidden_states)                       # [B, T, E]
        topk_values, topk_indices = torch.topk(router_logits, k=K, dim=-1)
        topk_weights = torch.softmax(topk_values.float(), dim=-1).to(hidden_states.dtype)

        # Run all experts; each returns (output, 0, 0) because Qwen3MLP does
        all_expert_outs = [self.experts[e](hidden_states)[0] for e in range(E)]
        expert_outputs = torch.stack(all_expert_outs, dim=2)             # [B, T, E, H]

        gather_idx = topk_indices.unsqueeze(-1).expand(B, T, K, H)
        topk_expert_outputs = expert_outputs.gather(dim=2, index=gather_idx)
        mixed_output = (topk_expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)

        concept_mat = getattr(self, "concept_mat", None)
        self.last_aux_loss = _moe_aux_loss(router_logits, topk_indices, E)
        self.last_statement_loss = _moe_statement_loss(router_logits, concept_mat)
        return mixed_output, self.last_aux_loss, self.last_statement_loss


# ---------------------------------------------------------------------------
# MoELoRALinear  –  LoRA adapter where each "adapter" is an MoE of low-rank
#                   experts.  The base weight stays frozen.
# ---------------------------------------------------------------------------

class MoELoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a Mixture-of-LoRA-Experts adapter.

    Forward:
        y = W·x  +  scaling · Σ_{i ∈ top-k} w_i · B_i·A_i·x

    where W is frozen, and {router, A_i, B_i} are the trainable parameters.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        num_experts: int,
        top_k: int,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.r = r
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = lora_alpha / r

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        ref = next(base_linear.parameters())

        # Frozen pretrained weight
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # Router: maps each token's representation to expert scores
        self.router = nn.Linear(in_features, num_experts, bias=False).to(
            device=ref.device, dtype=ref.dtype
        )
        nn.init.normal_(self.router.weight, std=0.02)

        # E experts, each is a low-rank A (down) + B (up) pair
        self.lora_A = nn.ModuleList([
            nn.Linear(in_features, r, bias=False).to(device=ref.device, dtype=ref.dtype)
            for _ in range(num_experts)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(r, out_features, bias=False).to(device=ref.device, dtype=ref.dtype)
            for _ in range(num_experts)
        ])

        # Standard LoRA init: A ~ kaiming, B = 0  →  adapter starts at 0
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i].weight)

        self.last_aux_loss = None
        self.last_statement_loss = None
        self.compute_statement_loss = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, in_features]  (attention/MLP hidden states)
        """
        base_out = self.base_linear(x)                                   # [B, T, out]

        # Router
        router_logits = self.router(x)                                   # [B, T, E]
        topk_values, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_values.float(), dim=-1).to(x.dtype)  # [B, T, K]

        # All expert outputs stacked: [B, T, E, out_features]
        # expert_outs = torch.stack(
        #     [self.lora_B[e](self.lora_A[e](x)) for e in range(self.num_experts)],
        #     dim=-2,
        # )
        A = torch.stack([self.lora_A[e].weight for e in range(self.num_experts)])  # [E, r, in]
        B = torch.stack([self.lora_B[e].weight for e in range(self.num_experts)])  # [E, out, r]
        intermediate = torch.einsum('bti,eri->bter', x, A)                    # [B, T, E, r]
        expert_outs  = torch.einsum('bter,eor->bteo', intermediate, B)        # [B, T, E, out]


        # Gather top-k experts
        out_features = expert_outs.shape[-1]
        gather_idx = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, out_features)
        topk_outs = expert_outs.gather(dim=-2, index=gather_idx)         # [B, T, K, out]

        lora_out = (topk_outs * topk_weights.unsqueeze(-1)).sum(dim=-2)  # [B, T, out]

        concept_mat = getattr(self, "concept_mat", None)
        self.last_aux_loss = _moe_aux_loss(router_logits, topk_indices, self.num_experts)
        if self.compute_statement_loss:
            self.last_statement_loss = _moe_statement_loss(router_logits, concept_mat)

        return base_out + lora_out * self.scaling


# ---------------------------------------------------------------------------
# inject_moe_lora  –  walks the model and swaps target nn.Linear modules
# ---------------------------------------------------------------------------

DEFAULT_LORA_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def inject_moe_lora(
    model: nn.Module,
    target_modules,          # set/list of leaf attribute names, e.g. {"q_proj", "v_proj"}
    r: int,
    num_experts: int,
    top_k: int,
    lora_alpha: float = 1.0,
) -> list:
    """Replace every nn.Linear whose attribute name is in *target_modules* with
    a MoELoRALinear.  Returns the list of injected MoELoRALinear instances."""
    target_modules = set(target_modules)
    moe_lora_layers = []

    # Collect replacements first to avoid modifying the module tree while iterating
    replacements = []
    for full_name, module in model.named_modules():
        leaf_name = full_name.split(".")[-1]
        if leaf_name in target_modules and isinstance(module, nn.Linear):
            replacements.append(full_name)

    for full_name in replacements:
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]
        base_linear = getattr(parent, leaf_name)

        # print(int(parts[2]))
        layer_no = int(parts[1]) if parts[1].isdigit() else int(parts[2])
        if layer_no  > 25:
            moe_lora = MoELoRALinear(
                base_linear=base_linear,
                r=r,
                num_experts=num_experts,
                top_k=top_k,
                lora_alpha=lora_alpha,
            )
            moe_lora.compute_statement_loss = True
            setattr(parent, leaf_name, moe_lora)
            moe_lora_layers.append(moe_lora)

    return moe_lora_layers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen3MoEConfig(Qwen3Config):
    """Qwen3Config extended with MoEMLP and MoELoRA fields."""

    model_type = "qwen3"

    def __init__(
        self,
        # MoEMLP (full block replacement)
        moe_layer_indices=None,
        num_experts_temp=4,
        top_k=1,
        router_aux_loss_weight=0.01,
        add_expert_mlp=False,
        # MoE-LoRA (low-rank adapters on linear layers)
        lora_r=0,
        lora_alpha=1.0,
        lora_num_experts=4,
        lora_top_k=1,
        lora_target_modules=None,
        add_moe_lora=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # MoEMLP
        self.moe_layer_indices = moe_layer_indices or []
        self.num_experts_temp = int(num_experts_temp)
        self.top_k = int(top_k)
        self.router_aux_loss_weight = float(router_aux_loss_weight)
        self.add_expert_mlp = add_expert_mlp
        # MoE-LoRA
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_num_experts = int(lora_num_experts)
        self.lora_top_k = int(lora_top_k)
        self.lora_target_modules = lora_target_modules or list(DEFAULT_LORA_TARGET_MODULES)
        self.add_moe_lora = add_moe_lora


# ---------------------------------------------------------------------------
# Model backbone (unchanged except MoEMLP integration)
# ---------------------------------------------------------------------------

class Qwen3MoEModel(Qwen3Model):
    config_class = Qwen3MoEConfig

    def __init__(self, config):
        super().__init__(config)

        moe_layer_indices = getattr(config, "moe_layer_indices", [])
        num_experts = getattr(config, "num_experts_temp", 4)
        self.top_k = getattr(config, "top_k", 1)
        self.moe_layers = []
        self.moe_lora_layers = []  # filled by _inject_lora()
        self.converted_layer_indices = []
        self.router_aux_loss_weight = 0
        self.num_experts_temp = 0

        if moe_layer_indices and getattr(config, "add_expert_mlp", False):
            hidden_size = config.hidden_size
            for layer_idx in moe_layer_indices:
                dense_mlp = self.layers[layer_idx].mlp
                self.layers[layer_idx].mlp = MoEMLP(
                    dense_mlp=dense_mlp,
                    hidden_size=hidden_size,
                    num_experts_temp=num_experts,
                    top_k=self.top_k,
                )
                self.moe_layers.append(self.layers[layer_idx].mlp)
                self.converted_layer_indices.append(layer_idx)
            print(f"Qwen3MoEModel: MoEMLP layers={self.converted_layer_indices}")

        if getattr(config, "add_moe_lora", False) and getattr(config, "lora_r", 0) > 0:
            layers = inject_moe_lora(
                model=self,
                target_modules=config.lora_target_modules,
                r=config.lora_r,
                num_experts=config.lora_num_experts,
                top_k=config.lora_top_k,
                lora_alpha=config.lora_alpha,
            )
            self.moe_lora_layers.extend(layers)
            print(f"Qwen3MoEModel: Injected MoE-LoRA into {len(layers)} linear layers ")

# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------

class MoECausalLM(Qwen3ForCausalLM):
    config_class = Qwen3MoEConfig

    def __init__(self, config):
        self._config = config
        super().__init__(config)

        self.model = Qwen3MoEModel(config)
        self.moe_layers = self.model.moe_layers
        self.converted_layer_indices = self.model.converted_layer_indices
        self.moe_lora_layers = self.model.moe_lora_layers          # filled by _inject_lora()
        self.lm_head.weight = self.model.embed_tokens.weight

        # Initialise from config so these attrs always exist on the instance,
        # regardless of whether MoEMLP or LoRA is active.
        self.router_aux_loss_weight = float(getattr(config, "router_aux_loss_weight", 0.0))
        self.num_experts_temp = int(getattr(config, "num_experts_temp", 0))
        self.top_k = int(getattr(config, "top_k", 1))

        # Inject LoRA immediately when the flag is set (needed for checkpoint loading:
        # the structure must exist before HF loads the state dict).
 

    # ---- config property (needed because Qwen3ForCausalLM assigns config) ----

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

    # ---- MoE-LoRA injection ----

    def _inject_lora(self, lora_r, lora_num_experts, lora_top_k, lora_alpha, lora_target_modules):
        """Inject MoELoRALinear into *this* model and record the layers."""
        layers = inject_moe_lora(
            model=self,
            target_modules=lora_target_modules,
            r=lora_r,
            num_experts=lora_num_experts,
            top_k=lora_top_k,
            lora_alpha=lora_alpha,
        )
        self.moe_lora_layers.extend(layers)
        # Keep config in sync so config.json reflects the actual structure.
        # vLLM reads config.json and calls MoECausalLM(config) directly —
        # add_moe_lora=True tells __init__ to rebuild LoRA before loading weights.
        self.config.add_moe_lora = True
        print(f"Injected MoE-LoRA into {len(layers)} linear layers "
              f"(r={lora_r}, experts={lora_num_experts}, top_k={lora_top_k})")

    # ---- from_pretrained ----

    @classmethod
    def _metadata_path(cls, model_name_or_path):
        local_dir = Path(str(model_name_or_path))
        if local_dir.is_dir():
            return local_dir / MOE_METADATA_FILENAME
        return None

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        # MoEMLP args
        moe_layer_indices=None,
        num_experts_temp=None,
        top_k=None,
        router_aux_loss_weight=None,
        # MoE-LoRA args
        lora_r=None,
        lora_alpha=None,
        lora_num_experts=None,
        lora_top_k=None,
        lora_target_modules=None,
        **kwargs,
    ):
        # torch_dtype / attn_implementation are model-loading kwargs, not config
        # fields — passing them to AutoConfig causes warnings/errors in some HF versions.
        _MODEL_ONLY_KWARGS = {"torch_dtype", "attn_implementation", "device_map"}
        config_kwargs = {k: v for k, v in kwargs.items() if k not in _MODEL_ONLY_KWARGS}
        config = AutoConfig.from_pretrained(path, **config_kwargs)

        # Apply explicit MoEMLP overrides
        if moe_layer_indices is not None:
            config.moe_layer_indices = moe_layer_indices
        if num_experts_temp is not None:
            config.num_experts_temp = num_experts_temp
        if top_k is not None:
            config.top_k = top_k
        if router_aux_loss_weight is not None:
            config.router_aux_loss_weight = router_aux_loss_weight

        # Apply explicit MoE-LoRA overrides
        if lora_r is not None:
            config.lora_r = lora_r
        if lora_alpha is not None:
            config.lora_alpha = lora_alpha
        if lora_num_experts is not None:
            config.lora_num_experts = lora_num_experts
        if lora_top_k is not None:
            config.lora_top_k = lora_top_k
        if lora_target_modules is not None:
            config.lora_target_modules = lora_target_modules

        # Check for saved MoE checkpoint metadata
        meta_path = Path(str(path)) / MOE_METADATA_FILENAME
        saved_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        loading_moe = saved_meta is not None and moe_layer_indices is None

        if loading_moe:
            # Restore full topology from saved metadata
            config.moe_layer_indices = saved_meta.get("moe_layer_indices", [])
            config.num_experts_temp = saved_meta.get("num_experts_temp", 0)
            config.top_k = saved_meta.get("top_k", 1)
            config.router_aux_loss_weight = saved_meta.get("router_aux_loss_weight", 0.0)
            if "lora_r" in saved_meta:
                config.lora_r = saved_meta["lora_r"]
                config.lora_alpha = saved_meta.get("lora_alpha", 1.0)
                config.lora_num_experts = saved_meta["lora_num_experts"]
                config.lora_top_k = saved_meta["lora_top_k"]
                config.lora_target_modules = saved_meta.get("lora_target_modules", list(DEFAULT_LORA_TARGET_MODULES))

            # Signal __init__ to build MoEMLP and LoRA *before* HF loads the
            # state dict, so saved weights land in the right modules.
            config.add_expert_mlp = bool(config.moe_layer_indices)
            config.add_moe_lora = getattr(config, "lora_r", 0) > 0

            state_dict = kwargs.pop("state_dict", None)
            if state_dict is None:
                # HF will call cls(config) internally → __init__ builds the full
                # structure, then loads weights into it.
                model = super(Qwen3ForCausalLM, cls).from_pretrained(path, config=config, **kwargs)
            else:
                model = cls(config)
                model.load_state_dict(state_dict, strict=False)
        else:
            # Fresh dense checkpoint: load weights first, then graft MoE on top.
            config.add_expert_mlp = False
            config.add_moe_lora = False
            model = super().from_pretrained(path, config=config, **kwargs)

            if getattr(config, "moe_layer_indices", []):
                model._replace_mlp_with_moe(
                    config.moe_layer_indices, config.num_experts_temp, config.top_k
                )

            # Inject MoE-LoRA after dense loading (fresh adapters start at zero)
            if getattr(config, "lora_r", 0) > 0:
                model._inject_lora(
                    lora_r=config.lora_r,
                    lora_num_experts=config.lora_num_experts,
                    lora_top_k=config.lora_top_k,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                )

        return model

    # ---- helpers for MoEMLP conversion (kept from original) ----

    def _get_transformer_blocks(self):
        if hasattr(self, "model") and hasattr(self.model, "layers"):
            return self.model.layers
        raise ValueError("Unsupported Qwen3 architecture: expected self.model.layers")

    def _get_mlp(self, block):
        for attr in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, attr):
                return attr, getattr(block, attr)
        raise ValueError(f"Cannot find MLP in block: {type(block)}.")

    def _replace_mlp_with_moe(self, moe_layer_indices, num_experts_temp, top_k):
        blocks = self._get_transformer_blocks()
        total_layers = len(blocks)
        for layer_idx in sorted(set(moe_layer_indices)):
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(f"Invalid layer index {layer_idx}. Valid range [0, {total_layers - 1}].")
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
        # Keep config in sync for vLLM: add_expert_mlp=True tells __init__ to
        # rebuild MoEMLP structure before loading weights from the checkpoint.
        self.config.add_expert_mlp = True

    # ---- forward ----

    def forward(self, *args, **kwargs):
        # Pop concept_mat so it is never forwarded to HF internals which don't
        # expect it (causes TypeError in newer transformers versions).
        concept_mat = kwargs.pop("concept_mat", None)

        # Propagate concept_mat to all MoE layers before the forward pass
        for layer in self.moe_layers:
            setattr(layer, "concept_mat", concept_mat)
        for layer in self.moe_lora_layers:
            setattr(layer, "concept_mat", concept_mat)

        outputs = super().forward(*args, **kwargs)

        if outputs.loss is not None and self.router_aux_loss_weight > 0.0:
            # --- MoEMLP losses (propagated via backbone return values) ---
            if self.moe_layers:
                aux_loss = outputs.loss1 / len(self.moe_layers)
                statement_loss = outputs.loss2 / len(self.moe_layers)
                outputs.loss = (
                    outputs.loss
                    + self.router_aux_loss_weight * aux_loss
                    + self.router_aux_loss_weight * 2 * statement_loss
                )

            # --- MoE-LoRA losses (collected after forward via stored attrs) ---
            if self.moe_lora_layers:
                lora_aux = [
                    l.last_aux_loss for l in self.moe_lora_layers if l.last_aux_loss is not None
                ]
                lora_stmt = [
                    l.last_statement_loss for l in self.moe_lora_layers if l.last_statement_loss is not None
                ]
                if lora_aux:
                    outputs.loss = outputs.loss + self.router_aux_loss_weight * torch.stack(lora_aux).mean()
                if lora_stmt:
                    outputs.loss = outputs.loss + self.router_aux_loss_weight * 2 * torch.stack(lora_stmt).mean()

        return outputs

    # ---- save ----

    def save_pretrained(self, save_directory, **kwargs):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is not None:
            kwargs["state_dict"] = {key: value for key, value in state_dict.items()}

        # Stamp auto_map so vLLM / HF AutoModel can locate our custom classes
        # when loading from this directory with trust_remote_code=True.
        self.config.architectures = ["MoECausalLM"]
        self.config.auto_map = {
            "AutoConfig": "model.Qwen3MoEConfig",
            "AutoModel": "model.Qwen3MoEModel",
            "AutoModelForCausalLM": "model.MoECausalLM",
        }

        result = self.model.save_pretrained(save_directory, **kwargs)

        # Persist all MoE topology so the checkpoint can be re-loaded
        metadata = {
            "moe_layer_indices": self.converted_layer_indices,
            "num_experts_temp": self.num_experts_temp,
            "top_k": self.top_k,
            "router_aux_loss_weight": self.router_aux_loss_weight,
        }
        if self.moe_lora_layers:
            cfg = self.config
            metadata.update({
                "lora_r": getattr(cfg, "lora_r", 0),
                "lora_alpha": getattr(cfg, "lora_alpha", 1.0),
                "lora_num_experts": getattr(cfg, "lora_num_experts", 4),
                "lora_top_k": getattr(cfg, "lora_top_k", 1),
                "lora_target_modules": getattr(cfg, "lora_target_modules", []),
            })
        with (Path(save_directory) / MOE_METADATA_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    explicit_moe = bool(args.moe_layer_indices)
    explicit_lora = bool(getattr(args, "lora_r", 0))

    model = MoECausalLM.from_pretrained(
        args.model_name,
        # MoEMLP
        moe_layer_indices=args.moe_layer_indices if explicit_moe else None,
        num_experts_temp=args.num_experts_temp if explicit_moe else None,
        top_k=args.moe_top_k if explicit_moe else None,
        router_aux_loss_weight=args.router_aux_loss_weight if (explicit_moe or explicit_lora) else None,
        # MoE-LoRA
        lora_r=args.lora_r if explicit_lora else None,
        lora_alpha=args.lora_alpha if explicit_lora else None,
        lora_num_experts=args.lora_num_experts if explicit_lora else None,
        lora_top_k=args.lora_top_k if explicit_lora else None,
        lora_target_modules=args.lora_target_modules if explicit_lora else None,
        # HF kwargs
        torch_dtype=getattr(args, "torch_dtype", torch.bfloat16),
        attn_implementation=getattr(args, "attn_implementation", None),
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing with use_reentrant=False.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer
