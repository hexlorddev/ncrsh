# Utility Functions

This module contains various utility functions and classes used throughout the ncrsh library.

## Core Utilities

### Checkpointing
- `checkpoint.py`: Save and load model checkpoints
- `logger.py`: Training metrics logging
- `serialization.py`: Model serialization utilities

### Data Processing
- `data_utils.py`: Data loading and processing functions
- `collate.py`: Custom collate functions for DataLoader
- `sampler.py`: Custom samplers for data loading

### Model Utilities
- `model_utils.py`: Model initialization and manipulation
- `parameter.py`: Parameter utilities
- `flops_counter.py`: FLOPs and parameter counting

### Visualization
- `visualize.py`: Model and training visualization
- `attention_viz.py`: Attention visualization
- `grad_cam.py`: Grad-CAM visualization

## Usage Examples

### Checkpointing
```python
from ncrsh.utils.checkpoint import CheckpointManager

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    model=model,
    optimizer=optimizer,
    checkpoint_dir="checkpoints",
    max_to_keep=3
)

# Save checkpoint
checkpoint_manager.save(epoch, metrics={"loss": loss.item()})

# Load latest checkpoint
checkpoint = checkpoint_manager.load()
```

### Data Processing
```python
from ncrsh.utils.data_utils import split_dataset, download_file

# Split dataset
train_set, val_set = split_dataset(dataset, [0.8, 0.2])

# Download file
file_path = download_file("https://example.com/data.zip", "data")
```

### Visualization
```python
from ncrsh.utils.visualize import plot_model, visualize_attention

# Plot model architecture
plot_model(model, input_size=(3, 224, 224), filename="model.png")

# Visualize attention
visualize_attention(
    attention_weights=attention,
    input_tokens=["the", "cat", "sat", "on", "the", "mat"],
    output_tokens=["le", "chat", "s'est", "assis", "sur", "le", "tapis"]
)
```

## Best Practices

1. **Error Handling**:
   - Use custom exceptions for better error messages
   - Validate inputs at function boundaries
   - Provide helpful error messages

2. **Documentation**:
   - Include docstrings with examples
   - Document expected input/output formats
   - Add type hints for better IDE support

3. **Testing**:
   - Write unit tests for all utility functions
   - Test edge cases and error conditions
   - Include property-based tests where applicable

4. **Performance**:
   - Optimize for common use cases
   - Use vectorized operations when possible
   - Add progress bars for long-running operations

## Adding New Utilities

1. Create a new Python file in the appropriate submodule
2. Add a descriptive docstring at the top of the file
3. Implement the functionality with type hints
4. Add unit tests in `tests/utils/`
5. Update this README if adding significant new features
