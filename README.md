# MoECoder

Minimal example for distributed LLM training with:
- `transformers`
- `accelerate`
- `deepspeed`
- custom training loop (no `Trainer`)

## Files
- `src/train.py`: custom language model training loop
- `src/model.py`: model definitions and MoE conversion logic
- `src/ds_config.json`: DeepSpeed config (ZeRO-2 + bf16)
- `src/accelerate_config.yaml`: Accelerate config for 2 GPUs

## Install
```bash
pip install torch transformers accelerate deepspeed datasets
```

## Train on 2 GPUs
```bash
accelerate launch --config_file src/accelerate_config.yaml src/train.py --do_train
```

## Evaluate only
```bash
accelerate launch --config_file src/accelerate_config.yaml src/train.py --do_eval
```

## Train + evaluate
```bash
accelerate launch --config_file src/accelerate_config.yaml src/train.py --do_train --do_eval
```

## Convert Selected MLP Layers To MoE
Example: convert transformer layers 4 and 8 from dense MLP to MoE.
Current implementation supports GPT-style models with `transformer.h[*].mlp`.

```bash
accelerate launch --config_file src/accelerate_config.yaml src/train.py \
  --do_train \
  --moe_layer_indices 4 8 \
  --num_experts 8 \
  --moe_top_k 2 \
  --router_aux_loss_weight 0.01
```
