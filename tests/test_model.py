import torch
import os
from ex2.models.model import MyAwesomeModel
import pytest

def test_model():
    model = MyAwesomeModel()
    x = torch.rand(1, 1, 28, 28)
    y = model(x)
    assert y.shape == torch.Size([1, 10]), "Output shape not matching expected format"
    assert y.dtype == torch.float32, "Output type not matching expected format"
    assert model.configure_optimizers() is not None, "Optimizer not configured"
    assert model.forward(x).shape == torch.Size([1, 10]), "Forward pass not working"

def test_error_msg():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
