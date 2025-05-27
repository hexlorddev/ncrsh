"""
Tests for model utilities.
"""
import os
import tempfile
import pytest
import numpy as np

import ncrsh
from ncrsh.nn import Sequential, Linear, ReLU, MSELoss
from ncrsh.optim import SGD
from ncrsh.utils.checkpoint import CheckpointManager, Logger

class SimpleModel(ncrsh.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class TestCheckpointManager:
    def test_checkpoint_save_load(self, tmp_path):
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # Initialize checkpoint manager
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            max_to_keep=2
        )
        
        # Generate some fake training data
        x = ncrsh.randn(5, 10)
        y = ncrsh.randn(5, 1)
        criterion = MSELoss()
        
        # Train for a few steps
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            manager.save(epoch, metrics={"loss": float(loss)})
        
        # Check that checkpoints were saved
        assert len(list(checkpoint_dir.glob("*.pth"))) > 0
        
        # Test loading
        loaded_checkpoint = manager.load()
        assert 'epoch' in loaded_checkpoint
        assert 'metrics' in loaded_checkpoint
        assert 'loss' in loaded_checkpoint['metrics']
    
    def test_checkpoint_best_model(self, tmp_path):
        model = SimpleModel()
        checkpoint_dir = tmp_path / "checkpoints_best"
        manager = CheckpointManager(
            model=model,
            checkpoint_dir=checkpoint_dir,
            save_best_only=True,
            metric_name="val_loss"
        )
        
        # Save with different metric values
        manager.save(0, metrics={"val_loss": 0.5})
        manager.save(1, metrics={"val_loss": 0.3})  # Better
        manager.save(2, metrics={"val_loss": 0.4})  # Worse, should not save
        
        # Only two checkpoints should exist: one for epoch 1 and the best model
        assert (checkpoint_dir / "model_best.pth").exists()
        assert len(list(checkpoint_dir.glob("*.pth"))) == 2

class TestLogger:
    def test_logger_basic(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = Logger(log_dir=log_dir)
        
        # Log some metrics
        for epoch in range(3):
            logger.log(epoch, {"loss": 1.0 / (epoch + 1), "accuracy": 0.1 * (epoch + 1)})
        
        # Check that log file was created
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) == 1
        
        # Check log content
        import json
        with open(log_files[0], 'r') as f:
            logs = json.load(f)
            
        assert len(logs) == 3
        assert 'loss' in logs[0]
        assert 'accuracy' in logs[0]
        assert logs[0]['epoch'] == 0
        assert logs[1]['epoch'] == 1
        assert logs[2]['epoch'] == 2
    
    def test_plot_metrics(self, tmp_path):
        log_dir = tmp_path / "logs_plot"
        logger = Logger(log_dir=log_dir)
        
        # Log some metrics
        for epoch in range(3):
            logger.log(epoch, {"loss": 1.0 / (epoch + 1), "accuracy": 0.1 * (epoch + 1)})
        
        # Test plotting
        plot_path = tmp_path / "metrics.png"
        logger.plot_metrics(save_path=plot_path, show=False)
        
        # Check that plot was saved
        assert plot_path.exists()

if __name__ == "__main__":
    pytest.main([__file__])
