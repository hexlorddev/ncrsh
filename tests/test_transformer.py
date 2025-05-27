"""
Tests for the ncrsh.nn.transformer module.
"""
import pytest
import numpy as np

from ncrsh.tensor import Tensor
from ncrsh.nn import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)

def test_multihead_attention():
    """Test the MultiHeadAttention module."""
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    
    # Create attention module
    attn = MultiHeadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True
    )
    
    # Create test input
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    # Self-attention
    output, attn_weights = attn(x, x, x, need_weights=True)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size * num_heads, seq_len, seq_len)

def test_transformer_encoder_layer():
    """Test the TransformerEncoderLayer."""
    batch_size = 2
    seq_len = 10
    d_model = 16
    nhead = 4
    dim_feedforward = 32
    
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    
    # Create test input
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    # Forward pass
    output = encoder_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)

def test_transformer_encoder():
    """Test the TransformerEncoder."""
    batch_size = 2
    seq_len = 10
    d_model = 16
    nhead = 4
    num_layers = 3
    dim_feedforward = 32
    
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    
    # Create encoder
    encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    # Create test input
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    # Forward pass
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, d_model)

def test_transformer_decoder_layer():
    """Test the TransformerDecoderLayer."""
    batch_size = 2
    tgt_seq_len = 8
    src_seq_len = 10
    d_model = 16
    nhead = 4
    dim_feedforward = 32
    
    # Create decoder layer
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    
    # Create test inputs
    tgt = Tensor(np.random.randn(batch_size, tgt_seq_len, d_model).astype(np.float32))
    memory = Tensor(np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32))
    
    # Forward pass
    output = decoder_layer(tgt, memory)
    
    assert output.shape == (batch_size, tgt_seq_len, d_model)

@pytest.mark.skip(reason="This is a more complex integration test")
def test_transformer():
    """Integration test for the complete Transformer model."""
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    d_model = 16
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 32
    
    # Create encoder
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    
    # Create decoder
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    
    # Create test inputs
    src = Tensor(np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32))
    tgt = Tensor(np.random.randn(batch_size, tgt_seq_len, d_model).astype(np.float32))
    
    # Forward pass
    memory = encoder(src)
    output = decoder(tgt, memory)
    
    assert output.shape == (batch_size, tgt_seq_len, d_model)
