# Model Checkpoints

This directory contains saved model checkpoints from training runs.

## Directory Structure

```
checkpoints/
├── model1/
│   ├── config.yaml
│   ├── checkpoint_001.pth
│   └── checkpoint_best.pth
└── model2/
    ├── config.yaml
    └── checkpoint_010.pth
```

## Guidelines

- **Naming Convention**: Use descriptive names for model directories
- **Checkpoint Files**: Save checkpoints with epoch numbers (e.g., `checkpoint_001.pth`)
- **Best Model**: Save the best model as `checkpoint_best.pth`
- **Configuration**: Include a `config.yaml` with training parameters
- **Logs**: Store training logs in the same directory as the checkpoints

## Best Practices

1. **Versioning**:
   - Include version numbers in directory names
   - Tag important model versions

2. **Metadata**:
   - Add a README.md in each model directory with:
     - Model architecture
     - Training parameters
     - Performance metrics
     - Training dataset

3. **Storage**:
   - Consider using Git LFS for large model files
   - Clean up old checkpoints regularly

## Example Directory Structure

```
checkpoints/
└── resnet50_imagenet_v1/
    ├── README.md
    ├── config.yaml
    ├── checkpoint_001.pth
    ├── checkpoint_010.pth
    ├── checkpoint_best.pth
    └── training_log.json
```

## Loading a Checkpoint

```python
from ncrsh.utils.checkpoint import load_checkpoint

# Load the latest checkpoint
checkpoint = load_checkpoint('checkpoints/resnet50_imagenet_v1')

# Access model state and other information
model_state = checkpoint['model_state']
epoch = checkpoint['epoch']
metrics = checkpoint['metrics']
```
