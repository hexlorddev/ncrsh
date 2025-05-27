"""
Base optimizer class for ncrsh.

This module defines the base optimizer class that all optimizers should inherit from.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

from ..tensor import Tensor

class Optimizer:
    """Base class for all optimizers.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        defaults: A dict containing default values of optimization parameters
    """
    
    def __init__(self, params, defaults: Dict):
        self.defaults = defaults
        self.param_groups: List[Dict[str, Any]] = []
        
        # Convert params to list if it's not already
        if not isinstance(params, (list, tuple)):
            params = [{'params': params}]
        
        # Initialize parameter groups
        for param_group in params:
            self.add_param_group(param_group)
    
    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'param_groups': self.param_groups,
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a :class:`dict`."""
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {param_mappings[id(k)]: v for k, v in self.state.items()}
        
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state.
        
        Args:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module and tensor
        state_dict = {k: v for k, v in state_dict.items()}
        
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']
        
        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups")
        
        # Update the state
        id_map = {old_id: p for old_id, p in zip(
            [id(p) for group in groups for p in group['params']],
            [p for group in groups for p in group['params']]
        )}
        
        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, Tensor):
                # Same as original tensor but on the right device.
                if param.is_cuda and not value.is_cuda:
                    value = value.cuda()
                elif not param.is_cuda and value.is_cuda:
                    value = value.cpu()
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return type(value)(cast(param, v) for v in value)
            else:
                return value
        
        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = {}
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v
        
        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        
        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Sets the gradients of all optimized :class:`Tensor`s to zero.
        
        Args:
            set_to_none: Instead of setting to zero, set the grads to None.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
    
    def add_param_group(self, param_group: Dict) -> None:
        """Add a param group to the :class:`Optimizer` s `param_groups`.
        
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        
        Args:
            param_group: Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError("param group must be a dict")
        
        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                          'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)
        
        for param in param_group['params']:
            if not isinstance(param, Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                              "but one of the params is " + type(param).__name__)
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")
        
        for name, default in self.defaults.items():
            if name not in param_group:
                param_group[name] = default
        
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")
        
        self.param_groups.append(param_group)
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        raise NotImplementedError
