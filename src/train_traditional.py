"""Train Softmax on a frozen feature split; select using internal validation only."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.data import load_features, load_manifest
from src.traditional.softmax import SoftmaxClassifier, fit_standardizer, loss_and_grad
from src.utils import classification_metrics, environment, new_output, read_json, write_json, write_predictions


def validate_config(config):
    expected = {'seed', 'learning_rate', 'epochs', 'batch_size', 'l2_values'}
    if set(config) != expected:
        raise ValueError(f'Configuration keys must be {sorted(expected)}')
    for key in ('seed', 'epochs', 'batch_size'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('Learning rate must be finite and positive')
    values = config['l2_values']
    if not isinstance(values, list) or not values or any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError('L2 values must be a nonempty list of finite nonnegative numbers')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate L2 values')


def train(data_dir, manifest_path, config, output):
    validate_config(config)
    manifest = load_manifest(manifest_path)
    x_train, y_train, _ = load_features(data_dir, manifest, 'train')
    x_val, y_val, val_rows = load_features(data_dir, manifest, 'internal_val')
    output = new_output(output)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'environment.json', environment())
    mean, scale = fit_standardizer(x_train)
    x_train = (x_train - mean) / scale
    x_val = (x_val - mean) / scale
    classes = manifest['selected_classes']
    started = time.perf_counter()
    candidates, history = [], []
    best_model, best_key = None, None
    for candidate, l2 in enumerate(config['l2_values']):
        # Identical initialization and minibatch order isolate the regularization effect.
        rng = np.random.default_rng(config['seed'])
        weight = np.zeros((x_train.shape[1], len(classes)), dtype=np.float64)
        bias = np.zeros(len(classes), dtype=np.float64)
        candidate_key, candidate_model, candidate_result = None, None, None
        for epoch in range(1, config['epochs'] + 1):
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), config['batch_size']):
                indices = order[start:start + config['batch_size']]
                _, dw, db = loss_and_grad(x_train[indices], y_train[indices], weight, bias, l2)
                weight -= config['learning_rate'] * dw
                bias -= config['learning_rate'] * db
            train_loss, _, _ = loss_and_grad(x_train, y_train, weight, bias)
            val_loss, _, _ = loss_and_grad(x_val, y_val, weight, bias)
            if not np.isfinite([train_loss, val_loss]).all():
                raise ValueError('Non-finite loss; inspect features and reduce the learning rate')
            train_acc = float(np.mean((x_train @ weight + bias).argmax(axis=1) == y_train))
            val_acc = float(np.mean((x_val @ weight + bias).argmax(axis=1) == y_val))
            result = {'candidate': candidate, 'l2': l2, 'epoch': epoch,
                      'train_loss': train_loss, 'internal_val_loss': val_loss,
                      'train_accuracy': train_acc, 'internal_val_accuracy': val_acc}
            history.append(result)
            key = (val_acc, -val_loss)
            if candidate_key is None or key > candidate_key:
                candidate_key, candidate_result = key, result.copy()
                metadata = {'version': 1, 'classes': classes, 'feature_key': manifest['feature_key'],
                            'manifest_sha256': manifest['manifest_sha256'],
                            'config': config, 'selection': candidate_result}
                candidate_model = SoftmaxClassifier(mean.copy(), scale.copy(), weight.copy(), bias.copy(), metadata)
        candidates.append(candidate_result)
        candidate_model.save(output / f'candidate-{candidate}.npz')
        if best_key is None or candidate_key > best_key:
            best_key, best_model = candidate_key, candidate_model
        print(f"l2={l2:g}: epoch={candidate_result['epoch']}, "
              f"internal_val_accuracy={candidate_result['internal_val_accuracy']:.4f}", flush=True)
    best_model.save(output / 'model.npz')
    with (output / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    prediction = (x_val @ best_model.weight + best_model.bias).argmax(axis=1)
    metrics = classification_metrics(y_val, prediction, len(classes))
    summary = {'task': 'traditional_10_class' if len(classes) == 10 else 'traditional_classification',
               'selection_rule': 'highest internal validation accuracy; lowest cross-entropy breaks ties; earliest entry breaks exact ties',
               'classes': classes, 'fit_samples': len(y_train), 'internal_val_samples': len(y_val),
               'feature_dimension': len(mean), 'manifest_sha256': manifest['manifest_sha256'],
               'candidates': candidates, 'selected': best_model.metadata['selection'],
               'internal_val': metrics, 'training_seconds': time.perf_counter() - started,
               'test_evaluated': False}
    write_json(output / 'summary.json', summary)
    write_predictions(output / 'internal_val_predictions.csv', val_rows, y_val, prediction, classes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/softmax.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = train(args.data_dir, args.manifest, read_json(args.config), args.output_dir)
    print(f"Saved model: {args.output_dir / 'model.npz'}")
    print(f"Internal validation: {summary['internal_val']['correct']}/{summary['internal_val']['total']}")


if __name__ == '__main__':
    main()
