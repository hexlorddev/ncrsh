"""
Base classes for all neural network modules in ncrsh.
"""
from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from ..tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    
    Your models should also subclass this class.
    """
    
    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = {}
        self._modules: Dict[str, Optional['Module']] = {}
    
    def forward(self, *input):
        """
        Define the forward pass of the module.
        
        Should be overridden by all subclasses.
        """
        raise NotImplementedError("Module [{}] is missing the required "
                              "forward function".format(self.__class__.__name__))
    
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, yields parameters of this module and all submodules.
                    Otherwise, yields only parameters that are direct members of this module.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Return an iterator over module parameters, yielding both the name and parameter.
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    
    def children(self) -> Iterator['Module']:
        """Return an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module
    
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over immediate children modules, yielding both the name and module."""
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
    
    def modules(self) -> Iterator['Module']:
        """Return an iterator over all modules in the network."""
        for name, module in self.named_modules():
            yield module
    
    def named_modules(self, memo: Optional[set] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over all modules in the network, yielding both the name and module."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
    
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        """Helper method for getting members with name."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
    
    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def __setattr__(self, name: str, value: Union[Tensor, 'Module', None]) -> None:
        """Set attributes while tracking parameters and submodules."""
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as parameter '{name}' "
                              "(torch.nn.Parameter or None expected)")
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{type(value).__name__}' as child module '{name}' "
                                  "(torch.nn.Module or None expected)")
                modules[name] = value
            else:
                object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str):
        """Get attributes, falling back to parameters and submodules."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
    
    def __delattr__(self, name):
        """Delete attributes, including parameters and submodules."""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)
    
    def register_parameter(self, name: str, param: Optional['Parameter']) -> None:
        """Register a parameter with the module."""
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        
        if param is None:
            self._parameters[name] = None
            return
        
        if not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' object to parameter '{name}' "
                         "(torch.nn.Parameter or None required)")
        
        if '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        
        if name in self._parameters:
            raise KeyError(f"parameter '{name}' already defined in {self.__class__.__name__}")
        
        self._parameters[name] = param


class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    
    Parameters are :class:`~Tensor` subclasses, that have a very special property
    when used with :class:`Module` s - when they're assigned as Module attributes
    they are automatically added to the list of its parameters, and will appear
    e.g. in :meth:`~Module.parameters` iterator. Assigning a Tensor doesn't have
    such effect. This is because one might want to cache some temporary state,
    like last hidden state of the RNN, in the model. If there was no such class as
    :class:`Parameter`, these temporaries would get registered too.
    """
    def __new__(cls, data: Optional[Tensor] = None, requires_grad: bool = True) -> 'Parameter':
        if data is None:
            data = Tensor([])
        return Tensor._make_subclass(cls, data, requires_grad)
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result
    
    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()
    
    def __reduce_ex__(self, proto):
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, dict(self.__dict__)))


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s
