import torch
import os
from tests import _PATH_DATA
import pytest

file_path = os.path.join(_PATH_DATA, "processed/train_images.pt")
@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():
    dataset = torch.load(os.path.join(file_path))
    assert len(dataset) == 25000
    for data, _ in dataset:
        assert data.shape ==  torch.Size([1, 28, 28]), "Data shape not matching expected formats"
        assert data.dtype == torch.float32, "Data type not matching expected format"
        assert data.max() <= 1.0, "Data not normalized"
        assert data.min() >= 0.0, "Data not normalized"
    assert len(dataset[:][1].unique()) == 10, "Not all labels are represented in the dataset"
