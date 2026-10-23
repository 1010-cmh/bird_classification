from itertools import combinations
import shutil

import numpy as np
import pytest

from src.data import audit_dataset, load_features, make_manifest
from src.evaluate import evaluate
from src.train_svm import train, validate_config
from src.traditional.io import load_model
from src.traditional.softmax import fit_standardizer
from src.traditional.svm import DEFAULT_SOLVER, SVMClassifier, fit_binary, kernel_matrix, vote_predictions
from src.utils import write_json


CONFIG = {'kernel': 'linear', 'gamma': None, 'C_values': [1., 0.1], 'solver': DEFAULT_SOLVER.copy()}


def test_separable_binary_problem_matches_analytic_solution():
    x = np.array([[-1.], [1.]])
    y = np.array([-1., 1.])
    model = fit_binary(x, y, c=2)
    np.testing.assert_allclose(model.dual_coefficients, [-0.5, 0.5], atol=1e-6)
    assert model.bias == pytest.approx(0, abs=1e-7)
    np.testing.assert_allclose(model.decision_function(np.array([[-3.], [0.], [2.]]), 'linear', None),
                               [-3, 0, 2], atol=1e-6)
    assert model.diagnostics['duality_gap'] < 1e-6
    assert model.diagnostics['free_support_vectors'] == 2


def test_no_free_support_vectors_uses_kkt_interval():
    x = np.array([[0.], [2.]])
    model = fit_binary(x, np.array([-1., 1.]), c=0.1)
    assert model.diagnostics['free_support_vectors'] == 0
    assert model.diagnostics['bias_method'] == 'kkt_interval'
    assert model.bias == pytest.approx(-0.2, abs=1e-6)
    np.testing.assert_allclose(model.decision_function(x, 'linear', None), [-0.2, 0.2], atol=1e-6)
    assert model.diagnostics['duality_gap'] < 1e-6


def test_nonseparable_identical_points_have_finite_solution():
    model = fit_binary(np.zeros((2, 3)), np.array([-1., 1.]), c=1)
    assert model.bias == pytest.approx(0, abs=1e-6)
    assert model.diagnostics['bias_method'] == 'kkt_interval'
    assert model.diagnostics['primal_objective'] == pytest.approx(2, abs=1e-6)
    assert model.diagnostics['kkt_residual'] < 1e-6


def test_rbf_solves_xor():
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = np.array([-1., 1., 1., -1.])
    model = fit_binary(x, y, c=10, kernel='rbf', gamma=1)
    np.testing.assert_array_equal(np.sign(model.decision_function(x, 'rbf', 1)), y)
    gram = kernel_matrix(x, x, 'rbf', 1)
    np.testing.assert_allclose(np.diag(gram), 1)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12


def test_nonconvergence_is_rejected():
    with pytest.raises(ValueError, match='did not converge'):
        fit_binary(np.array([[-1.], [1.]]), np.array([-1., 1.]), c=2,
                   solver_options={'maxiters': 1})


def test_vote_ties_and_majority_priority():
    pairs = list(combinations(range(3), 2))
    # Each class gets one vote; larger positive margin in (1, 2) favors class 2.
    assert vote_predictions(np.array([[1., -1., 10.]]), pairs, 3).tolist() == [2]
    assert vote_predictions(np.array([[1., -1., 1.]]), pairs, 3).tolist() == [0]
    # Class 0 wins all its matches narrowly; class 3 cannot override that majority.
    margins = np.array([[-1e-5, -1e-5, -1e-5, 100., 100., 100.]])
    assert vote_predictions(margins, list(combinations(range(4), 2)), 4).tolist() == [0]


@pytest.mark.parametrize('kernel,gamma', [('linear', None), ('rbf', 0.5)])
def test_multiclass_round_trip(tmp_path, kernel, gamma):
    x = np.array([[-2., 0.], [-3., 0.], [2., 0.], [3., 0.], [0., 2.], [0., 3.]])
    y = np.array([0, 0, 1, 1, 2, 2])
    mean, scale = fit_standardizer(x)
    model = SVMClassifier.fit(x, y, mean, scale, ['a', 'b', 'c'], 10, kernel, gamma, DEFAULT_SOLVER, {})
    np.testing.assert_array_equal(model.predict(x), y)
    path = tmp_path / 'svm.npz'
    model.save(path)
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    assert loaded.metadata['kernel'] == kernel
    assert len(loaded.models) == 3


def test_training_and_evaluation_do_not_read_test_features(dataset, tmp_path):
    summary, records = audit_dataset(dataset, expected_classes=3)
    manifest = make_manifest(summary, records, num_classes=3)
    path = tmp_path / 'manifest.json'
    write_json(path, manifest)
    x_train, _, _ = load_features(dataset, manifest, 'train')
    shutil.rmtree(dataset / 'val')
    result = train(dataset, path, CONFIG, tmp_path / 'train')
    assert result['test_evaluated'] is False
    assert result['selected']['C'] == 0.1
    assert result['internal_val']['accuracy'] == 1
    model = load_model(tmp_path / 'train/model.npz')
    np.testing.assert_allclose(model.mean, x_train.mean(axis=0))
    metrics = evaluate(dataset, path, tmp_path / 'train/model.npz', 'internal_val', tmp_path / 'evaluation')
    assert metrics['accuracy'] == result['internal_val']['accuracy']
    assert (tmp_path / 'evaluation/predictions.csv').read_bytes() == (tmp_path / 'train/internal_val_predictions.csv').read_bytes()


@pytest.mark.parametrize('update', [{'C_values': []}, {'C_values': [0]}, {'C_values': [1, 1]},
                                    {'C_values': [float('nan')]}, {'kernel': 'other'},
                                    {'kernel': 'rbf', 'gamma': None}, {'gamma': 0.1},
                                    {'solver': {**DEFAULT_SOLVER, 'maxiters': 0}}])
def test_invalid_svm_configuration(update):
    with pytest.raises(ValueError):
        validate_config({**CONFIG, **update})
