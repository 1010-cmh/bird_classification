"""RGB image loading for all classes in a frozen dataset manifest."""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils import dataset_path, file_hash


def image_tensor(image, image_size):
    """Preserve aspect ratio, center-pad, then normalize pixels to [-1, 1]."""
    image = image.convert('RGB')
    ratio = min(image_size / image.width, image_size / image.height)
    width = min(image_size, max(1, round(image.width * ratio)))
    height = min(image_size, max(1, round(image.height * ratio)))
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (image_size, image_size), (128, 128, 128))
    canvas.paste(resized, ((image_size - width) // 2, (image_size - height) // 2))
    pixels = np.array(canvas, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class BirdImageDataset(Dataset):
    def __init__(self, root, manifest, split, image_size):
        if split not in ('train', 'internal_val', 'test') or image_size < 16:
            raise ValueError('Invalid image split or resolution')
        self.rows = manifest['splits'][split]
        self.class_to_idx = manifest['class_to_idx']
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.image_size = image_size
        self.paths = []
        for row in self.rows:
            path = dataset_path(root, row['image'])
            if file_hash(path) != row['image_sha256']:
                raise ValueError(f"Image changed since dataset audit: {row['image']}")
            self.paths.append(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = image_tensor(image, self.image_size)
        return tensor, self.class_to_idx[self.rows[index]['class']]
