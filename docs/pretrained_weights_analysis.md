# Analysis: Pretrained Weights Loading in DEIMKit

## The Issue

When you use the `--pretrained` flag in the training script, it does NOT load the full DEIM pretrained weights. Instead, it only configures the backbone to use ImageNet pretrained weights.

## Code Trace

### 1. In `scripts/azure_train.py`:
```python
# Configure model
conf = configure_model(
    conf, 
    num_queries=args.num_queries,
    freeze_at=args.freeze_at,
    pretrained=args.pretrained  # This is the --pretrained flag
)
```

### 2. In `src/deimkit/model.py` - `configure_model()`:
```python
if pretrained is not None:
    update_setting("backbone", pretrained, backbone_type, "pretrained")
```

This only sets `yaml_cfg.HGNetv2.pretrained = True`, which tells the HGNetv2 backbone to load ImageNet weights, NOT DEIM weights!

### 3. The Missing Link

The `get_model_checkpoint()` function in `model.py` can download DEIM pretrained weights:
```python
MODEL_CHECKPOINT_URLS = {
    "deim_hgnetv2_n": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_n_coco_160e.pth",
    "deim_hgnetv2_s": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_s_coco_120e.pth",
    # ... etc
}
```

But this is ONLY used by the `Predictor` class for inference, NOT by the `Trainer` class!

### 4. Current Behavior

- `--pretrained` flag → Only loads ImageNet weights for the backbone
- `--checkpoint` flag → Resumes training from a checkpoint (your own checkpoint)
- No flag exists to load the official DEIM pretrained weights!

## The Solution

To properly fine-tune from DEIM pretrained weights, you need to:

### Option 1: Download and Use as Checkpoint
```bash
# First, download the pretrained weights
python -c "from deimkit.model import get_model_checkpoint; print(get_model_checkpoint('deim_hgnetv2_s'))"

# This will download to ./checkpoints/deim_hgnetv2_s.pth
# Then use it as a checkpoint:
python scripts/azure_train.py \
    --train-ann-file /path/to/train.json \
    --train-img-folder /path/to/train/images \
    --val-ann-file /path/to/val.json \
    --val-img-folder /path/to/val/images \
    --model-name deim_hgnetv2_s \
    --checkpoint ./checkpoints/deim_hgnetv2_s.pth \  # Load DEIM pretrained weights
    --num-classes 10 \
    --epochs 50
```

### Option 2: Modify the Training Script

Add this to `azure_train.py` after creating the trainer:

```python
# Create trainer
print("Initializing trainer...")
trainer = Trainer(conf, callbacks=callbacks)

# Load DEIM pretrained weights if pretrained flag is set
if args.pretrained and not args.checkpoint:
    from deimkit.model import get_model_checkpoint
    pretrained_checkpoint = get_model_checkpoint(args.model_name)
    print(f"Loading DEIM pretrained weights from: {pretrained_checkpoint}")
    trainer.load_checkpoint(pretrained_checkpoint)
```

## Summary

The current `--pretrained` flag is misleading:
- It does NOT load DEIM pretrained weights
- It only enables ImageNet pretrained weights for the backbone
- To use DEIM pretrained weights, you must use them as a checkpoint

This is a significant issue because users expect `--pretrained` to load the full model's pretrained weights, not just the backbone's ImageNet weights.

## Verification

You can verify this by checking the model's performance:
- With `--pretrained` only: Poor initial mAP (model starts from scratch except backbone)
- With DEIM checkpoint: Good initial mAP (model starts from COCO-trained weights)
