import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    rng = np.random.default_rng(3)
    for source, count in [('train', 12), ('val', 3)]:
        for label in range(3):
            folder = root / source / f'{label:03d}.bird'
            folder.mkdir(parents=True)
            for i in range(count):
                pixels = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f'{i:03d}.jpg')
                vector = rng.normal(0, 0.04, 6)
                vector[label] += 4
                vector[-1] = 1
                torch.save(torch.tensor(vector, dtype=torch.float32).unsqueeze(0), folder / f'{i:03d}.pt')
    return root
