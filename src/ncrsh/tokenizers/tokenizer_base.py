"""
Base tokenizer class for ncrsh.

This module defines the base interface for all tokenizers in ncrsh.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import os
import json

class TokenizerBase(ABC):
    """
    Abstract base class for all tokenizers in ncrsh.
    
    This class defines the interface that all tokenizers must implement.
    """
    
    def __init__(self, **kwargs):
        """Initialize the tokenizer with optional configuration."""
        self.added_tokens: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.pad_token: Optional[str] = None
        self.unk_token: Optional[str] = None
        self.bos_token: Optional[str] = None
        self.eos_token: Optional[str] = None
        self.cls_token: Optional[str] = None
        self.sep_token: Optional[str] = None
        self.mask_token: Optional[str] = None
        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.cls_token_id: Optional[int] = None
        self.sep_token_id: Optional[int] = None
        self.mask_token_id: Optional[int] = None
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        raise NotImplementedError
    
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        
        Args:
            text: The input string to tokenize.
            **kwargs: Additional arguments to pass to the tokenizer.
            
        Returns:
            A list of tokens.
        """
        pass
    
    @abstractmethod
    def encode(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode a string or a list of strings into token IDs.
        
        Args:
            text: The input text or list of texts to encode.
            text_pair: Optional second sequence to encode (for sequence pairs).
            add_special_tokens: Whether to add special tokens.
            padding: Whether to pad the sequences.
            truncation: Whether to truncate the sequences.
            max_length: Maximum length of the returned list of tokens.
            return_tensors: If set to 'pt' or 'tf', returns PyTorch or TensorFlow tensors.
            
        Returns:
            A list of token IDs or a list of lists of token IDs.
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        """
        Decode a token ID or a list of token IDs back to a string.
        
        Args:
            token_ids: The token ID or list of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in the output.
            clean_up_tokenization_spaces: Whether to clean up the tokenization spaces.
            
        Returns:
            The decoded string.
        """
        pass
    
    @abstractmethod
    def save_pretrained(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save the tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the tokenizer to.
            **kwargs: Additional arguments to pass to the save method.
            
        Returns:
            A tuple of file paths that were saved.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the tokenizer configuration
        tokenizer_config = {
            "tokenizer_class": self.__class__.__name__,
            "added_tokens": self.added_tokens,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
        }
        
        # Add any additional configuration
        tokenizer_config.update(self.get_config())
        
        # Save the configuration
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        return (config_path,)
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> 'TokenizerBase':
        """
        Instantiate a tokenizer from a pretrained model or directory.
        
        Args:
            pretrained_model_name_or_path: Name or path to the pretrained tokenizer.
            **kwargs: Additional arguments to pass to the tokenizer initialization.
            
        Returns:
            A tokenizer instance.
        """
        pass
    
    def get_config(self) -> Dict:
        """
        Get the tokenizer configuration.
        
        Returns:
            A dictionary containing the tokenizer configuration.
        """
        return {}
    
    def add_tokens(self, new_tokens: Union[str, List[str]]) -> int:
        """
        Add new tokens to the tokenizer.
        
        Args:
            new_tokens: A single token or a list of tokens to add.
            
        Returns:
            The number of tokens that were added.
        """
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        
        added = 0
        for token in new_tokens:
            if token in self.added_tokens:
                continue
            
            token_id = self.vocab_size + len(self.added_tokens)
            self.added_tokens[token] = token_id
            self.added_tokens_decoder[token_id] = token
            added += 1
        
        return added
    
    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, int]]) -> int:
        """
        Add special tokens to the tokenizer.
        
        Args:
            special_tokens_dict: A dictionary of special tokens to add.
                Keys should be in the list of predefined special tokens ('bos_token', 'eos_token', etc.).
                
        Returns:
            The number of tokens that were added.
        """
        added = 0
        
        for key, value in special_tokens_dict.items():
            if value is None:
                continue
                
            if key == 'additional_special_tokens':
                added += self.add_tokens(value)
            elif hasattr(self, key):
                setattr(self, key, value)
                if value not in self.added_tokens:
                    token_id = self.vocab_size + len(self.added_tokens)
                    self.added_tokens[value] = token_id
                    self.added_tokens_decoder[token_id] = value
                    added += 1
        
        return added
