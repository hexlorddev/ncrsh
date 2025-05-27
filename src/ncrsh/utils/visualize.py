"""
Model Visualization Utilities
---------------------------
This module provides utilities for visualizing neural network models.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import warnings

import numpy as np

import ncrsh
from ncrsh.nn import Module
from ncrsh.tensor import Tensor

# Try to import graphviz, but don't fail if not available
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    warnings.warn(
        "Graphviz is not installed. Install with 'pip install graphviz' "
        "to enable model visualization."
    )

def make_dot(
    output: Tensor,
    params: Optional[Dict[str, Any]] = None,
    show_attrs: bool = False,
    show_saved: bool = False,
) -> 'Digraph':
    """Produce Graphviz representation of the autograd graph.
    
    This function is similar to torchviz.make_dot but works with ncrsh tensors.
    """
    if not HAS_GRAPHVIZ:
        raise ImportError("Graphviz is not installed. Install with 'pip install graphviz'")
    
    from graphviz import Digraph
    
    param_map = {id(p): name for name, p in params.items()} if params else {}
    
    node_attr = {
        'style': 'filled',
        'shape': 'box',
        'align': 'left',
        'fontsize': '12',
        'ranksep': '0.1',
        'height': '0.2',
        'fontname': 'monospace',
    }
    
    dot = Digraph(node_attr=node_attr, graph_attr={'size': '12,12'})
    seen = set()
    
    def add_nodes(var):
        if var in seen:
            return
        
        seen.add(var)
        
        if hasattr(var, 'next_functions'):
            for u in var.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
        
        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                dot.edge(str(id(t)), str(id(var)), style='dashed')
                add_nodes(t)
        
        # Format the node text
        if hasattr(var, 'variable'):
            u = var.variable
            name = param_map.get(id(u), '')
            node_name = f'{name}\n{tuple(u.size())}'
            dot.node(str(id(var)), node_name, fillcolor='lightblue')
        else:
            # For operations
            node_name = type(var).__name__
            if hasattr(var, 'grad_fn'):
                node_name = str(var.grad_fn)
                node_name = node_name.split('Backward')[0]  # Clean up the name
            
            # Add shape info if available
            if hasattr(var, 'shape'):
                node_name += f'\n{tuple(var.shape)}'
            
            dot.node(str(id(var)), node_name, fillcolor='#f0f0f0')
    
    # Add the output node
    dot.node(str(id(output)), 'Output', fillcolor='darkolivegreen1', style='filled')
    
    # Add all nodes
    add_nodes(output.grad_fn)
    
    return dot


def plot_model(
    model: Module,
    input_size: Tuple[int, ...],
    filename: Optional[Union[str, Path]] = None,
    format: str = 'png',
    show_attrs: bool = False,
    show_saved: bool = False,
    **kwargs
) -> Optional['Digraph']:
    """Plot a model's computation graph.
    
    Args:
        model: The model to visualize
        input_size: Input tensor shape (batch dimension is optional)
        filename: File to save the visualization to. If None, the graph is not saved.
        format: Output format (e.g., 'png', 'pdf', 'svg')
        show_attrs: Whether to show attributes in the graph
        show_saved: Whether to show saved tensors in the graph
        **kwargs: Additional arguments to pass to make_dot
        
    Returns:
        The Digraph object if graphviz is installed, else None
    """
    if not HAS_GRAPHVIZ:
        warnings.warn("Graphviz is not installed. Install with 'pip install graphviz'")
        return None
    
    # Create a dummy input
    x = ncrsh.randn(*input_size, requires_grad=True)
    
    # Forward pass
    try:
        y = model(x)
    except Exception as e:
        raise RuntimeError(f"Failed to run model forward pass: {e}")
    
    # Create the graph
    try:
        # Get parameters for the model
        params = dict(model.named_parameters())
        
        # Create the graph
        dot = make_dot(y, params=params, show_attrs=show_attrs, show_saved=show_saved)
        
        # Save to file if filename is provided
        if filename is not None:
            filename = Path(filename).with_suffix(f'.{format}')
            filename.parent.mkdir(parents=True, exist_ok=True)
            dot.render(filename, format=format, cleanup=True)
            print(f"Model graph saved to {filename}")
        
        return dot
    except Exception as e:
        warnings.warn(f"Failed to generate model graph: {e}")
        return None


def count_parameters(module: Module) -> Dict[str, int]:
    """Count the number of parameters in a module.
    
    Args:
        module: The module to count parameters for
        
    Returns:
        Dictionary with 'total_params', 'trainable_params', and 'non_trainable_params' counts
    """
    total_params = 0
    trainable_params = 0
    
    for p in module.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def model_summary(
    model: Module,
    input_size: Tuple[int, ...],
    device: Optional[Union[str, ncrsh.device]] = None,
    dtypes: Optional[List[ncrsh.dtype]] = None
) -> str:
    """Print a summary of the model architecture and parameters.
    
    Args:
        model: The model to summarize
        input_size: Input tensor shape (batch dimension is optional)
        device: Device to run the model on
        dtypes: List of input dtypes (one for each input)
        
    Returns:
        String summary of the model
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
    
    if dtypes is None:
        dtypes = [ncrsh.float32] * len(input_size) if isinstance(input_size, (list, tuple)) else [ncrsh.float32]
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {
                "input_shape": [tuple(i.size()) for i in input] if isinstance(input, (list, tuple)) else [tuple(input.size())],
                "output_shape": [tuple(o.size()) for o in output] if isinstance(output, (list, tuple)) else [tuple(output.size())],
                "nb_params": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
        
        if (not isinstance(module, ncrsh.nn.Sequential) and
                not isinstance(module, ncrsh.nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))
    
    # Check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [ncrsh.randn(2, *in_size, device=device, dtype=dt) 
             for in_size, dt in zip(input_size, dtypes)]
    else:
        x = ncrsh.randn(2, *input_size, device=device, dtype=dtypes[0])
    
    # Create properties
    summary = {}
    hooks = []
    
    # Register hook
    model.apply(register_hook)
    
    # Make a forward pass
    model(x) if not isinstance(x, (list, tuple)) else model(*x)
    
    # Remove these hooks
    for h in hooks:
        h.remove()
    
    # Generate summary string
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    summary_str = f"{'-'*80}\n"
    line_new = f"{'Layer (type)':<25}  {'Output Shape':<25} {'Param #':<15}"
    summary_str += line_new + '\n' + '=' * 80 + '\n'
    
    for layer in summary:
        # Input shape
        line_new = f"{layer:<25}  {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:,}"
        total_params += summary[layer]['nb_params']
        
        # Output shape
        output_shape = summary[layer]['output_shape']
        total_output += np.prod([x for xs in output_shape for x in xs])
        
        # Add to summary
        summary_str += line_new + '\n'
    
    # Add total parameters
    summary_str += '=' * 80 + '\n'
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
    
    # Add input size
    summary_str += '-' * 80 + '\n'
    summary_str += f"Input size (MB): {np.prod(input_size) * 4. / (1024 ** 2):.2f}\n"
    
    # Add forward/backward pass size
    total_output_size = 2 * total_output * 4. / (1024 ** 2)  # x2 for gradients
    summary_str += f"Forward/backward pass size (MB): {total_output_size:.2f}\n"
    
    # Add params size
    params_size = total_params * 4. / (1024 ** 2)
    summary_str += f"Params size (MB): {params_size:.2f}\n"
    
    # Add estimated total size
    summary_str += f"Estimated Total Size (MB): {np.prod(input_size) * 4. / (1024 ** 2) + total_output_size + params_size:.2f}\n"
    summary_str += '-' * 80 + '\n'
    
    return summary_str


def visualize_attention(
    attention_weights: Tensor,
    input_tokens: Optional[List[str]] = None,
    output_tokens: Optional[List[str]] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """Visualize attention weights.
    
    Args:
        attention_weights: Attention weights tensor of shape (target_len, source_len)
        input_tokens: List of input token strings (optional)
        output_tokens: List of output token strings (optional)
        cmap: Matplotlib colormap to use
        figsize: Figure size (width, height)
        save_path: Path to save the figure to (optional)
        show: Whether to display the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy if not already
    if hasattr(attention_weights, 'numpy'):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(
        attention_weights,
        cmap=cmap,
        square=True,
        cbar=True,
        xticklabels=input_tokens if input_tokens is not None else False,
        yticklabels=output_tokens if output_tokens is not None else False,
    )
    
    # Set labels
    if input_tokens is not None:
        plt.xticks(rotation=45, ha='right')
    if output_tokens is not None:
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Attention visualization saved to {save_path}")
    
    # Show figure
    if show:
        plt.show()
    
    plt.close()
