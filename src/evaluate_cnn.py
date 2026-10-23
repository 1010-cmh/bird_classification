"""Evaluate an image classifier without refitting or selecting a model."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.cnn_runtime import load_inference_model, resolve_device
from src.data import load_manifest
from src.image_data import BirdImageDataset
from src.train_cnn import run_epoch
from src.utils import classification_metrics, file_hash, new_output, write_json, write_predictions


def evaluate(data_dir, manifest_path, checkpoint_path, split, output_dir, device_name='auto'):
    device = resolve_device(device_name)
    manifest = load_manifest(manifest_path)
    model, checkpoint = load_inference_model(checkpoint_path, device)
    classes = sorted(manifest['class_to_idx'], key=manifest['class_to_idx'].get)
    if checkpoint['manifest_sha256'] != manifest['manifest_sha256'] or checkpoint['classes'] != classes:
        raise ValueError('CNN checkpoint and manifest do not match')
    config = checkpoint['config']
    dataset = BirdImageDataset(data_dir, manifest, split, config['image_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    measured, labels, predictions = run_epoch(model, loader, device)
    metrics = {**classification_metrics(labels, predictions, len(classes)),
               'loss': measured['loss'], 'split': split, 'classes': classes,
               'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_kind': checkpoint['kind'],
               'checkpoint_sha256': file_hash(checkpoint_path),
               'manifest_sha256': manifest['manifest_sha256'], 'device': str(device)}
    output = new_output(output_dir)
    write_json(output / 'metrics.json', metrics)
    write_predictions(output / 'predictions.csv', dataset.rows, labels, predictions, classes)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['internal_val', 'test'], default='internal_val')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    metrics = evaluate(args.data_dir, args.manifest, args.checkpoint, args.split, args.output_dir, args.device)
    print(f"{args.split}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")


if __name__ == '__main__':
    main()
