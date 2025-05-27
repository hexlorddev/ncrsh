"""
Text Classification Example
--------------------------
This example demonstrates how to train a simple LSTM for text classification
using the ncrsh library.
"""
import os
import numpy as np
from pathlib import Path
import random
from collections import Counter

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import ncrsh
from ncrsh.data import DataLoader, Dataset
from ncrsh.nn import (
    Module, Sequential, Embedding, LSTM, Linear, 
    Dropout, CrossEntropyLoss, ReLU
)
from ncrsh.optim import Adam
from ncrsh.tensor import Tensor

# Set random seed for reproducibility
ncrsh.manual_seed(42)
random.seed(42)

class TextClassificationModel(Module):
    """A simple LSTM model for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = Dropout(dropout)
        self.fc = Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM returns: output, (hidden_state, cell_state)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden state for classification
        hidden = self.dropout(hidden[-1])  # Take the last layer's hidden state
        return self.fc(hidden)

class TextDataset(Dataset):
    """A simple text classification dataset."""
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Create vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
        
        # Add special tokens
        self.pad_idx = 0
        self.unk_idx = 1
        self.vocab_size = len(self.vocab) + 2  # +2 for PAD and UNK
        
        # Tokenize texts
        self.tokenized_texts = [self._tokenize(text) for text in texts]
    
    def _build_vocab(self, texts, min_freq=2):
        # Tokenize all texts
        tokens = []
        for text in texts:
            tokens.extend(text.lower().split())
        
        # Count token frequencies
        counter = Counter(tokens)
        
        # Create vocabulary with tokens appearing at least min_freq times
        vocab = {}
        for token, count in counter.items():
            if count >= min_freq:
                vocab[token] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text):
        # Simple tokenization (split by whitespace)
        tokens = text.lower().split()
        
        # Convert tokens to indices, using unk_idx for unknown tokens
        indices = [self.vocab.get(token, self.unk_idx) + 2 for token in tokens]  # +2 because 0 and 1 are reserved
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.pad_idx] * (self.max_length - len(indices))
            
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return Tensor(self.tokenized_texts[idx]), Tensor([self.labels[idx]])

def train_one_epoch(model, dataloader, criterion, optimizer, device=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if device:
            inputs = inputs.to(device).long()
            targets = targets.to(device).squeeze(1)  # Remove extra dimension
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device=None):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with ncrsh.no_grad():
        for inputs, targets in dataloader:
            if device:
                inputs = inputs.to(device).long()
                targets = targets.to(device).squeeze(1)  # Remove extra dimension
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def create_dummy_data(num_samples=1000, num_classes=5):
    """Create dummy text classification data."""
    # Generate some dummy text data
    vocab = [f"word_{i}" for i in range(100)]
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Randomly select a label
        label = random.randint(0, num_classes - 1)
        
        # Generate a random text with 5-15 words
        text_length = random.randint(5, 15)
        text = " ".join(random.choices(vocab, k=text_length))
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels, num_classes

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 2
    embedding_dim = 100
    hidden_dim = 128
    num_layers = 2
    dropout = 0.5
    
    # Check for CUDA
    device = ncrsh.device('cuda' if ncrsh.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data (in a real scenario, you would load real data)
    print("Creating dummy data...")
    train_texts, train_labels, num_classes = create_dummy_data(num_samples=2000)
    val_texts, val_labels, _ = create_dummy_data(num_samples=500)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(
        val_texts, 
        val_labels, 
        vocab=train_dataset.vocab,  # Use the same vocabulary as training
        max_length=train_dataset.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model, loss, and optimizer
    vocab_size = train_dataset.vocab_size
    model = TextClassificationModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model architecture:\n{model}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\nTraining complete!")
    
    # Save the trained model and vocabulary
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/text_classifier.pth"
    ncrsh.save({
        'model_state_dict': model.state_dict(),
        'vocab': train_dataset.vocab,
        'max_length': train_dataset.max_length,
        'num_classes': num_classes,
        'model_params': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': num_classes,
            'num_layers': num_layers,
            'dropout': dropout
        }
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
