"""Train and select one-versus-one SVMs on the fixed internal validation split."""

import argparse
import time
from pathlib import Path

import numpy as np

from src.data import load_features, load_manifest
from src.traditional.softmax import fit_standardizer
from src.traditional.svm import DEFAULT_SOLVER, SVMClassifier
from src.utils import classification_metrics, environment, new_output, read_json, write_json, write_predictions


def validate_config(config):
    if set(config) != {'kernel', 'gamma', 'C_values', 'solver'}:
        raise ValueError('Configuration must contain kernel, gamma, C_values and solver')
    if config['kernel'] not in ('linear', 'rbf'):
        raise ValueError('Kernel must be linear or rbf')
    gamma = config['gamma']
    if config['kernel'] == 'linear' and gamma is not None:
        raise ValueError('Linear kernel requires gamma=null')
    if config['kernel'] == 'rbf' and (gamma is None or not np.isfinite(gamma) or gamma <= 0):
        raise ValueError('RBF gamma must be finite and positive')
    values = config['C_values']
    if (not isinstance(values, list) or not values
            or any(isinstance(c, bool) or not np.isfinite(c) or c <= 0 for c in values)
            or len(set(values)) != len(values)):
        raise ValueError('C_values must contain distinct finite positive numbers')
    options = config['solver']
    if not isinstance(options, dict) or set(options) != set(DEFAULT_SOLVER):
        raise ValueError(f'Solver options must be {sorted(DEFAULT_SOLVER)}')
    for name, value in options.items():
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid solver option: {name}')
    if not isinstance(options['maxiters'], int):
        raise ValueError('Solver maxiters must be an integer')


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
    classes = manifest['selected_classes']
    candidates, diagnostics = [], []
    best_model, best_key = None, None
    started = time.perf_counter()
    for candidate, c in enumerate(config['C_values']):
        begin = time.perf_counter()
        metadata = {'feature_key': manifest['feature_key'],
                    'manifest_sha256': manifest['manifest_sha256'], 'config': config}
        model = SVMClassifier.fit(x_train, y_train, mean, scale, classes, c,
                                  config['kernel'], config['gamma'], config['solver'], metadata)
        train_metrics = classification_metrics(y_train, model.predict(x_train), len(classes))
        val_metrics = classification_metrics(y_val, model.predict(x_val), len(classes))
        pairs = [{'negative_class': classes[a], 'positive_class': classes[b], **binary.diagnostics}
                 for (a, b), binary in zip(model.pairs, model.models, strict=True)]
        result = {'candidate': candidate, 'C': c, 'kernel': config['kernel'], 'gamma': config['gamma'],
                  'train_accuracy': train_metrics['accuracy'], 'internal_val_accuracy': val_metrics['accuracy'],
                  'support_vector_appearances': sum(p['support_vectors'] for p in pairs),
                  'max_kkt_residual': max(p['kkt_residual'] for p in pairs),
                  'fit_and_validation_seconds': time.perf_counter() - begin}
        candidates.append(result)
        diagnostics.append({'candidate': candidate, 'C': c, 'pairs': pairs})
        model.metadata['selection'] = result
        model.save(output / f'candidate-{candidate}.npz')
        key = (val_metrics['accuracy'], -c)
        if best_key is None or key > best_key:
            best_key, best_model = key, model
        print(f"C={c:g}: internal_val_accuracy={val_metrics['accuracy']:.4f}, "
              f"max_kkt_residual={result['max_kkt_residual']:.2e}", flush=True)
    best_model.save(output / 'model.npz')
    prediction = best_model.predict(x_val)
    summary = {'task': 'traditional_10_class' if len(classes) == 10 else 'traditional_classification',
               'model_type': 'svm', 'classes': classes, 'fit_samples': len(y_train),
               'internal_val_samples': len(y_val), 'feature_dimension': len(mean),
               'manifest_sha256': manifest['manifest_sha256'],
               'selection_rule': 'highest internal validation accuracy; smallest C breaks ties',
               'vote_rule': 'pairwise majority; bounded decision margins break ties; lowest class index breaks exact ties',
               'candidates': candidates, 'selected': best_model.metadata['selection'],
               'internal_val': classification_metrics(y_val, prediction, len(classes)),
               'training_seconds': time.perf_counter() - started, 'test_evaluated': False}
    write_json(output / 'summary.json', summary)
    write_json(output / 'solver_diagnostics.json', diagnostics)
    write_predictions(output / 'internal_val_predictions.csv', val_rows, y_val, prediction, classes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/svm-linear.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = train(args.data_dir, args.manifest, read_json(args.config), args.output_dir)
    print(f"Saved model: {args.output_dir / 'model.npz'}")
    print(f"Internal validation: {summary['internal_val']['correct']}/{summary['internal_val']['total']}")


if __name__ == '__main__':
    main()
