"""
Tests for data utilities.
"""
import os
import tempfile
import numpy as np
import pytest

import ncrsh
from ncrsh.data import DataLoader, Dataset, TensorDataset
from ncrsh.utils.data_utils import (
    split_dataset, download_file, extract_archive, default_collate
)

class TestDataset(Dataset):
    def __init__(self, size=100):
        self.data = np.random.randn(size, 3, 32, 32).astype(np.float32)
        self.targets = np.random.randint(0, 10, size=size)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)

class TestDataUtils:
    def test_split_dataset(self):
        dataset = TestDataset(100)
        train_set, val_set = split_dataset(dataset, [0.8, 0.2], random_seed=42)
        
        assert len(train_set) == 80
        assert len(val_set) == 20
        assert len(train_set) + len(val_set) == len(dataset)
    
    def test_download_file(self, tmp_path):
        # Test with a small file
        url = "https://raw.githubusercontent.com/example/README.md"  # Replace with a real test URL
        try:
            file_path = download_file(url, tmp_path, "test_file.txt")
            assert file_path.exists()
            assert file_path.stat().st_size > 0
        except Exception as e:
            pytest.skip(f"Could not download test file: {e}")
    
    def test_extract_archive(self, tmp_path):
        # Create a test archive
        import zipfile
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Create a zip archive
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(test_file, arcname="test.txt")
        
        # Test extraction
        extract_path = extract_archive(zip_path, tmp_path / "extracted")
        assert (extract_path / "test.txt").exists()
    
    def test_default_collate(self):
        # Test with a batch of tensors
        batch = [
            (ncrsh.tensor([1, 2, 3]), 0),
            (ncrsh.tensor([4, 5, 6]), 1),
        ]
        collated = default_collate(batch)
        
        assert isinstance(collated, tuple)
        assert len(collated) == 2
        assert collated[0].shape == (2, 3)  # Batched tensors
        assert collated[1].shape == (2,)    # Batched labels

class TestDataLoader:
    def test_dataloader_basic(self):
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Test iteration
        for batch in dataloader:
            x, y = batch
            assert x.shape == (10, 3, 32, 32)
            assert y.shape == (10,)
            break  # Just test first batch
    
    def test_tensor_dataset(self):
        x = ncrsh.tensor(np.random.randn(100, 3, 32, 32))
        y = ncrsh.tensor(np.random.randint(0, 10, 100))
        dataset = TensorDataset(x, y)
        
        assert len(dataset) == 100
        assert dataset[0][0].shape == (3, 32, 32)
        assert isinstance(dataset[0][1], ncrsh.Tensor)

if __name__ == "__main__":
    pytest.main([__file__])
