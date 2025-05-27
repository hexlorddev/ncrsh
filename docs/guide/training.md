# Training Models in ncrsh

This guide covers the process of training deep learning models using ncrsh, including setting up training loops, using optimizers, and monitoring training progress.

## Table of Contents

1. [Basic Training Loop](#basic-training-loop)
2. [Using Optimizers](#using-optimizers)
3. [Learning Rate Scheduling](#learning-rate-scheduling)
4. [Monitoring Training](#monitoring-training)
5. [Checkpointing](#checkpointing)
6. [Distributed Training](#distributed-training)
7. [Tips and Best Practices](#tips-and-best-practices)

## Basic Training Loop

Here's a minimal training loop in ncrsh:

```python
import ncrsh
from ncrsh.nn import CrossEntropyLoss
from ncrsh.optim import SGD

def train(model, train_loader, val_loader, num_epochs=10, lr=0.01):
    # Initialize loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with ncrsh.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Accuracy: {100 * correct / total:.2f}%')
        print('-' * 50)
```

## Using Optimizers

ncrsh provides various optimizers in the `ncrsh.optim` module:

```python
from ncrsh.optim import SGD, Adam, AdamW, RMSprop

# SGD with momentum
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam optimizer
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# Adam with weight decay
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)
```

## Learning Rate Scheduling

### Step-based Scheduling

```python
from ncrsh.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

# Step every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Multiple milestones
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# In training loop
for epoch in range(num_epochs):
    # Training code...
    scheduler.step()
```

### Cosine Annealing

```python
from ncrsh.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
```

## Monitoring Training

### Using TensorBoard

```python
from ncrsh.utils.tensorboard import SummaryWriter

# Initialize writer
writer = SummaryWriter('runs/experiment_1')

# In training loop
for epoch in range(num_epochs):
    # Training...
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    
    # Add histograms of weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()
```

## Checkpointing

### Saving and Loading Models

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Distributed Training

### Data Parallel

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
```

### Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Initialize process group
setup(rank, world_size)


# Create model and move to GPU
model = model.to(rank)
ddp_model = DDP(model, device_ids=[rank])

# In training loop
for epoch in range(num_epochs):
    # Training code...
    pass

cleanup()
```

## Tips and Best Practices

1. **Gradient Clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Gradient Accumulation**:
   ```python
   accumulation_steps = 4
   
   for i, (inputs, targets) in enumerate(train_loader):
       outputs = model(inputs)
       loss = criterion(outputs, targets) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Mixed Precision Training**:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   
   for inputs, targets in train_loader:
       optimizer.zero_grad()
       
       with torch.cuda.amp.autocast():
           outputs = model(inputs)
           loss = criterion(outputs, targets)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

4. **Early Stopping**:
   ```python
   patience = 5
   best_loss = float('inf')
   counter = 0
   
   for epoch in range(num_epochs):
       # Training and validation...
       
       if val_loss < best_loss:
           best_loss = val_loss
           counter = 0
           # Save best model
           torch.save(model.state_dict(), 'best_model.pth')
       else:
           counter += 1
           if counter >= patience:
               print("Early stopping!")
               break
   ```

5. **Learning Rate Finder**:
   ```python
   from torch_lr_finder import LRFinder
   
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
   
   lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
   lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
   lr_finder.plot()
   lr_finder.reset()
   ```
