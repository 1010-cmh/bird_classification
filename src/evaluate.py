"""Evaluate a saved traditional model against its frozen manifest."""

import argparse
from pathlib import Path

from src.data import load_features, load_manifest
from src.traditional.io import load_model
from src.utils import classification_metrics, file_hash, new_output, write_json, write_predictions


def evaluate(data_dir, manifest_path, model_path, split, output_dir):
    manifest = load_manifest(manifest_path)
    model = load_model(model_path)
    if (model.metadata['manifest_sha256'] != manifest['manifest_sha256']
            or model.metadata['classes'] != manifest['selected_classes']
            or model.metadata['feature_key'] != manifest['feature_key']):
        raise ValueError('Model and manifest do not match')
    x, y, rows = load_features(data_dir, manifest, split)
    prediction = model.predict(x)
    metrics = classification_metrics(y, prediction, len(manifest['selected_classes']))
    metrics.update({'split': split, 'classes': manifest['selected_classes'],
                    'manifest_sha256': manifest['manifest_sha256'],
                    'model_sha256': file_hash(model_path)})
    output = new_output(output_dir)
    write_json(output / 'metrics.json', metrics)
    write_predictions(output / 'predictions.csv', rows, y, prediction, manifest['selected_classes'])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--split', choices=['internal_val', 'test'], default='internal_val')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate(args.data_dir, args.manifest, args.model, args.split, args.output_dir)
    print(f"{args.split}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")


if __name__ == '__main__':
    main()
