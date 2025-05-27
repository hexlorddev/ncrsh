"""
Tests for visualization utilities.
"""
import os
import sys
import pytest
import numpy as np
import matplotlib

# Use non-interactive backend for testing
matplotlib.use('Agg')

import ncrsh
from ncrsh.nn import Sequential, Linear, ReLU
from ncrsh.utils.visualize import plot_model, visualize_attention, model_summary

class SimpleModel(ncrsh.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 1, bias=True)
        )
    
    def forward(self, x):
        return self.net(x)

class TestVisualization:
    def test_plot_model(self, tmp_path):
        """Test model plotting functionality."""
        model = SimpleModel()
        
        # Test with save
        output_path = tmp_path / "model_plot.png"
        plot_model(
            model=model,
            input_size=(10,),
            filename=output_path,
            show=False
        )
        
        # Check if file was created (if graphviz is available)
        try:
            from graphviz import Digraph
            assert output_path.exists()
        except ImportError:
            pytest.skip("Graphviz not installed")
    
    def test_visualize_attention(self, tmp_path):
        """Test attention visualization."""
        # Create dummy attention weights
        attention = ncrsh.tensor([
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.6, 0.3, 0.1]
        ])
        
        # Test with save
        output_path = tmp_path / "attention.png"
        visualize_attention(
            attention_weights=attention,
            input_tokens=["a", "b", "c"],
            output_tokens=["x", "y", "z"],
            save_path=output_path,
            show=False
        )
        
        assert output_path.exists()
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = SimpleModel()
        summary = model_summary(model, input_size=(10,))
        
        # Check if summary contains expected information
        assert "Layer (type)" in summary
        assert "Output Shape" in summary
        assert "Param #" in summary
        assert "Total params" in summary
        assert "Trainable params" in summary
        
        # Check if parameters are counted correctly
        expected_params = (10 * 20 + 20) + (20 * 1 + 1)  # weights + biases
        assert f"{expected_params}" in summary

if __name__ == "__main__":
    pytest.main([__file__])
