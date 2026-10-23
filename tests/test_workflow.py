import csv
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.data import load_features, load_manifest
from src.evaluate import evaluate
from src.train_traditional import train, validate_config
from src.traditional.softmax import SoftmaxClassifier
from src.utils import read_json


CONFIG = {'seed': 42, 'learning_rate': 0.05, 'epochs': 30,
          'batch_size': 7, 'l2_values': [0, 0.01]}


def test_prepare_train_restore_and_evaluate_without_test_data(dataset, tmp_path):
    manifest_path = tmp_path / 'split.json'
    command = [sys.executable, '-m', 'src.prepare_data', '--data-dir', str(dataset),
               '--expected-classes', '3', '--num-classes', '3', '--output', str(manifest_path),
               '--audit-output', str(tmp_path / 'audit.json')]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert subprocess.run(command, capture_output=True).returncode != 0
    manifest = load_manifest(manifest_path)
    x_train, _, _ = load_features(dataset, manifest, 'train')
    # Removing official test files proves training and validation do not read them.
    shutil.rmtree(dataset / 'val')
    first = train(dataset, manifest_path, CONFIG, tmp_path / 'run1')
    second = train(dataset, manifest_path, CONFIG, tmp_path / 'run2')
    assert first['test_evaluated'] is False
    assert first['internal_val']['accuracy'] == 1.0
    assert first['selected'] == second['selected']
    model = SoftmaxClassifier.load(tmp_path / 'run1/model.npz')
    np.testing.assert_allclose(model.mean, x_train.mean(axis=0))
    other = SoftmaxClassifier.load(tmp_path / 'run2/model.npz')
    np.testing.assert_array_equal(model.weight, other.weight)
    metrics = evaluate(dataset, manifest_path, tmp_path / 'run1/model.npz',
                       'internal_val', tmp_path / 'evaluation')
    assert metrics['accuracy'] == first['internal_val']['accuracy']
    with (tmp_path / 'run1/history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    assert float(history[-1]['train_loss']) < float(history[0]['train_loss'])
    training_predictions = (tmp_path / 'run1/internal_val_predictions.csv').read_bytes()
    assert training_predictions == (tmp_path / 'evaluation/predictions.csv').read_bytes()
    assert read_json(tmp_path / 'run1/manifest.json') == manifest
    with pytest.raises(FileExistsError):
        train(dataset, manifest_path, CONFIG, tmp_path / 'run1')


@pytest.mark.parametrize('update', [{'epochs': 0}, {'batch_size': -1}, {'seed': -1},
                                    {'learning_rate': float('nan')}, {'l2_values': [-0.1]},
                                    {'l2_values': []}, {'l2_values': [0, 0]}])
def test_invalid_training_config(update):
    with pytest.raises(ValueError):
        validate_config({**CONFIG, **update})
