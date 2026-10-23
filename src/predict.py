"""Predict the bird class of a single RGB or grayscale image."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from src.cnn_runtime import load_inference_model, resolve_device
from src.image_data import image_tensor


def predict(image_path, checkpoint_path, device_name='auto', top_k=5):
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    device = resolve_device(device_name)
    model, checkpoint = load_inference_model(checkpoint_path, device)
    with Image.open(image_path) as image:
        inputs = image_tensor(image, checkpoint['config']['image_size']).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(checkpoint['classes'])))
    return {'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_kind': checkpoint['kind'],
            'predictions': [{'label': int(i), 'class': checkpoint['classes'][i], 'probability': float(p)}
                            for i, p in zip(indices.cpu().tolist(), values.cpu().tolist(), strict=True)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.checkpoint, args.device, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
