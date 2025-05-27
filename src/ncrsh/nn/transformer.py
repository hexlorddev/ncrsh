"""
Transformer architecture implementation for ncrsh.

This module contains the implementation of the Transformer model architecture
introduced in "Attention Is All You Need" by Vaswani et al., along with
optimizations and improvements.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union, Callable

import math
from ..tensor import Tensor
from .modules import Module, Parameter
from .linear import Linear
from .dropout import Dropout
from .activation import GELU


class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism as described in "Attention Is All You Need".
    
    This module computes multi-head attention with optional masking and dropout.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
    ) -> None:
        """
        Initialize the MultiHeadAttention module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Whether to add bias to input/output projections
            add_bias_kv: If specified, adds bias to the key and value sequences
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences
            kdim: Total number of features for keys
            vdim: Total number of features for values
            batch_first: If True, input tensors are expected to be (batch, seq, feature)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Initialize projection layers
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(Tensor(3 * embed_dim, embed_dim))
        else:
            self.q_proj_weight = Parameter(Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(Tensor(embed_dim, self.vdim))
        
        if bias:
            self.in_proj_bias = Parameter(Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = Parameter(Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        if self._qkv_same_embed_dim:
            # Use Xavier initialization for the weight
            stdv = 1.0 / math.sqrt(self.embed_dim)
            self.in_proj_weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.embed_dim)
            self.q_proj_weight.data.uniform_(-stdv, stdv)
            self.k_proj_weight.data.uniform_(-stdv, stdv)
            self.v_proj_weight.data.uniform_(-stdv, stdv)
        
        if self.in_proj_bias is not None:
            self.in_proj_bias.data.zero_()
            self.out_proj.bias.data.zero_()
        
        if self.bias_k is not None:
            stdv = 1.0 / math.sqrt(self.embed_dim)
            self.bias_k.data.uniform_(-stdv, stdv)
            self.bias_v.data.uniform_(-stdv, stdv)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query embeddings of shape (L, N, E) if batch_first=False, else (N, L, E)
            key: Key embeddings of shape (S, N, E_k) if batch_first=False, else (N, S, E_k)
            value: Value embeddings of shape (S, N, E_v) if batch_first=False, else (N, S, E_v)
            key_padding_mask: If specified, a mask of shape (N, S) indicating which elements to ignore
            need_weights: If True, returns attention weights
            attn_mask: 2D or 3D mask that prevents attention to certain positions
            
        Returns:
            attn_output: Attention outputs of shape (L, N, E) if batch_first=False, else (N, L, E)
            attn_weights: Attention weights if need_weights=True, else None
        """
        # Implementation would go here
        pass


class TransformerEncoderLayer(Module):
    """
    A single layer of the transformer encoder.
    
    This layer consists of multi-head self-attention followed by a feedforward network.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = 'relu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        """
        Initialize the transformer encoder layer.
        
        Args:
            d_model: The number of expected features in the input
            nhead: The number of heads in the multi-head attention models
            dim_feedforward: The dimension of the feedforward network model
            dropout: The dropout value
            activation: The activation function of the intermediate layer
            layer_norm_eps: The epsilon value for layer normalization
            batch_first: If True, input tensors are expected to be (batch, seq, feature)
            norm_first: If True, layer norm is done prior to attention and feedforward operations
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = lambda x: x.relu()
        elif activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = activation
        
        self.norm_first = norm_first
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the transformer encoder layer.
        
        Args:
            src: The sequence to the encoder layer of shape (S, N, E) if batch_first=False, else (N, S, E)
            src_mask: The mask for the src sequence
            src_key_padding_mask: The mask for the src keys per batch
            
        Returns:
            Output tensor of the same shape as src
        """
        # Implementation would go here
        pass


class TransformerDecoderLayer(Module):
    """
    A single layer of the transformer decoder.
    
    This layer consists of multi-head self-attention, multi-head cross-attention,
    and a feedforward network.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = 'relu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        """
        Initialize the transformer decoder layer.
        
        Args:
            d_model: The number of expected features in the input
            nhead: The number of heads in the multi-head attention models
            dim_feedforward: The dimension of the feedforward network model
            dropout: The dropout value
            activation: The activation function of the intermediate layer
            layer_norm_eps: The epsilon value for layer normalization
            batch_first: If True, input tensors are expected to be (batch, seq, feature)
            norm_first: If True, layer norm is done prior to attention and feedforward operations
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = lambda x: x.relu()
        elif activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = activation
        
        self.norm_first = norm_first
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the transformer decoder layer.
        
        Args:
            tgt: The sequence to the decoder layer of shape (T, N, E) if batch_first=False, else (N, T, E)
            memory: The sequence from the last layer of the encoder of shape (S, N, E) if batch_first=False, else (N, S, E)
            tgt_mask: The mask for the tgt sequence
            memory_mask: The mask for the memory sequence
            tgt_key_padding_mask: The mask for the tgt keys per batch
            memory_key_padding_mask: The mask for the memory keys per batch
            
        Returns:
            Output tensor of the same shape as tgt
        """
        # Implementation would go here
        pass


class TransformerEncoder(Module):
    """
    Transformer encoder consisting of multiple TransformerEncoderLayer layers.
    """
    
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        """
        Initialize the transformer encoder.
        
        Args:
            encoder_layer: An instance of the TransformerEncoderLayer class
            num_layers: The number of sub-encoder-layers in the encoder
            norm: The layer normalization component (optional)
        """
        super().__init__()
        self.layers = [encoder_layer] * num_layers
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the transformer encoder.
        
        Args:
            src: The sequence to the encoder of shape (S, N, E) if batch_first=False, else (N, S, E)
            mask: The mask for the src sequence
            src_key_padding_mask: The mask for the src keys per batch
            
        Returns:
            Output tensor of shape (S, N, E) if batch_first=False, else (N, S, E)
        """
        output = src
        
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output


class TransformerDecoder(Module):
    """
    Transformer decoder consisting of multiple TransformerDecoderLayer layers.
    """
    
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        """
        Initialize the transformer decoder.
        
        Args:
            decoder_layer: An instance of the TransformerDecoderLayer class
            num_layers: The number of sub-decoder-layers in the decoder
            norm: The layer normalization component (optional)
        """
        super().__init__()
        self.layers = [decoder_layer] * num_layers
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the transformer decoder.
        
        Args:
            tgt: The sequence to the decoder of shape (T, N, E) if batch_first=False, else (N, T, E)
            memory: The sequence from the last layer of the encoder of shape (S, N, E) if batch_first=False, else (N, S, E)
            tgt_mask: The mask for the tgt sequence
            memory_mask: The mask for the memory sequence
            tgt_key_padding_mask: The mask for the tgt keys per batch
            memory_key_padding_mask: The mask for the memory keys per batch
            
        Returns:
            Output tensor of shape (T, N, E) if batch_first=False, else (N, T, E)
        """
        output = tgt
        
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output
