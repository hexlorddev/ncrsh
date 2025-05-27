"""
Configuration for pytest.
"""
import pytest
import numpy as np
import ncrsh

# Set random seed for reproducibility
np.random.seed(42)
ncrsh.manual_seed(42)

@pytest.fixture(scope="session")
def device():
    """Fixture to get the available device."""
    return ncrsh.device("cuda" if ncrsh.cuda.is_available() else "cpu")

@pytest.fixture
test_tensor():
    """Create a test tensor."""
    return ncrsh.tensor(np.random.randn(3, 3))

@pytest.fixture
test_model():
    """Create a simple test model."""
    model = ncrsh.nn.Sequential(
        ncrsh.nn.Linear(10, 20),
        ncrsh.nn.ReLU(),
        ncrsh.nn.Linear(20, 2)
    )
    return model

@pytest.fixture
test_dataset():
    """Create a test dataset."""
    class TestDataset(ncrsh.data.Dataset):
        def __init__(self, size=100):
            self.data = ncrsh.tensor(np.random.randn(size, 3, 32, 32).astype(np.float32))
            self.targets = ncrsh.tensor(np.random.randint(0, 10, size=size))
        
        def __getitem__(self, index):
            return self.data[index], self.targets[index]
        
        def __len__(self):
            return len(self.data)
    
    return TestDataset()

@pytest.fixture
test_dataloader(test_dataset):
    """Create a test dataloader."""
    return ncrsh.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
