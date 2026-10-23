"""Soft-margin SVM dual formulation and one-versus-one classification."""

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from cvxopt import matrix, solvers


DEFAULT_SOLVER = {'abstol': 1e-12, 'reltol': 1e-10, 'feastol': 1e-10, 'maxiters': 100}


def kernel_matrix(x, z, kernel='linear', gamma=None):
    if kernel == 'linear':
        return x @ z.T
    if kernel == 'rbf':
        if gamma is None or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('RBF gamma must be finite and positive')
        squared = np.sum(x * x, axis=1)[:, None] + np.sum(z * z, axis=1) - 2 * x @ z.T
        return np.exp(-gamma * np.maximum(squared, 0))
    raise ValueError(f'Unknown kernel: {kernel}')


def recover_bias(y, response, alpha, c, tolerance):
    """Use free support vectors or the interval implied by the KKT conditions."""
    free = (alpha > tolerance) & (alpha < c - tolerance)
    residual = y - response
    if np.any(free):
        return float(residual[free].mean()), 'free_support_vectors'
    lower = residual[((y == 1) & (alpha <= tolerance)) | ((y == -1) & (alpha >= c - tolerance))]
    upper = residual[((y == -1) & (alpha <= tolerance)) | ((y == 1) & (alpha >= c - tolerance))]
    if not len(lower) or not len(upper):
        raise ValueError('Cannot recover a finite bias interval')
    lo, hi = float(lower.max()), float(upper.min())
    if lo > hi + 1e-4:
        raise ValueError(f'Inconsistent KKT bias interval: [{lo}, {hi}]')
    return (lo + hi) / 2, 'kkt_interval'


@dataclass
class BinarySVM:
    support_vectors: np.ndarray
    dual_coefficients: np.ndarray
    bias: float
    diagnostics: dict

    def decision_function(self, x, kernel, gamma):
        if kernel == 'linear':
            weight = self.dual_coefficients @ self.support_vectors
            return x @ weight + self.bias
        return kernel_matrix(x, self.support_vectors, kernel, gamma) @ self.dual_coefficients + self.bias


def fit_binary(x, y, c, kernel='linear', gamma=None, solver_options=None):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),)
            or set(np.unique(y)) != {-1., 1.} or not np.isfinite(x).all()):
        raise ValueError('Binary training requires finite features and both -1/+1 labels')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('C must be finite and positive')
    options = {**DEFAULT_SOLVER, **(solver_options or {}), 'show_progress': False}
    gram = kernel_matrix(x, x, kernel, gamma)
    q = gram * y[:, None] * y[None, :]
    q = (q + q.T) / 2
    n = len(y)
    # CVXOPT minimizes 1/2 alpha^T Q alpha - 1^T alpha under box/equality constraints.
    result = solvers.qp(matrix(q), matrix(-np.ones(n)),
                        matrix(np.vstack((-np.eye(n), np.eye(n)))),
                        matrix(np.concatenate((np.zeros(n), np.full(n, c)))),
                        matrix(y.reshape(1, -1)), matrix(0.0), options=options)
    if result['status'] != 'optimal':
        raise ValueError(f"SVM quadratic program did not converge: {result['status']}")
    raw_alpha = np.asarray(result['x']).reshape(-1)
    equality_residual = float(abs(raw_alpha @ y))
    bound_violation = float(max(0, -raw_alpha.min(), raw_alpha.max() - c))
    if not np.isfinite(raw_alpha).all() or max(equality_residual, bound_violation) > 1e-6:
        raise ValueError('SVM dual solution violates feasibility tolerances')
    alpha = np.clip(raw_alpha, 0, c)
    tolerance = min(1e-7, c * 1e-5)
    alpha[alpha <= tolerance] = 0
    support = alpha > 0
    if not np.any(support):
        raise ValueError('No support vectors found')
    response = gram @ (alpha * y)
    bias, bias_method = recover_bias(y, response, alpha, c, tolerance)
    margin = y * (response + bias)
    free = (alpha > tolerance) & (alpha < c - tolerance)
    at_lower, at_upper = alpha <= tolerance, alpha >= c - tolerance
    violation = np.zeros(n)
    violation[free] = np.abs(margin[free] - 1)
    violation[at_lower] = np.maximum(1 - margin[at_lower], 0)
    violation[at_upper] = np.maximum(margin[at_upper] - 1, 0)
    kkt_residual = float(violation.max())
    norm_squared = float((alpha * y) @ response)
    primal = 0.5 * norm_squared + c * np.maximum(1 - margin, 0).sum()
    dual = alpha.sum() - 0.5 * norm_squared
    diagnostics = {'status': result['status'], 'iterations': int(result['iterations']),
                   'samples': n, 'support_vectors': int(support.sum()),
                   'free_support_vectors': int(free.sum()), 'bias_method': bias_method,
                   'equality_residual': equality_residual, 'bound_violation': bound_violation,
                   'effective_equality_residual': float(abs(alpha @ y)),
                   'kkt_residual': kkt_residual, 'primal_objective': float(primal),
                   'dual_objective': float(dual), 'duality_gap': float(primal - dual)}
    if kkt_residual > 1e-3 or diagnostics['effective_equality_residual'] > 1e-5:
        raise ValueError(f'SVM support-vector solution failed KKT checks: {diagnostics}')
    return BinarySVM(x[support].copy(), (alpha[support] * y[support]).copy(), bias, diagnostics)


def vote_predictions(margins, pairs, class_count):
    votes = np.zeros((len(margins), class_count), dtype=np.int64)
    confidence = np.zeros_like(votes, dtype=np.float64)
    rows = np.arange(len(margins))
    for column, (negative, positive) in enumerate(pairs):
        margin = margins[:, column]
        winner = np.where(margin > 0, positive, negative)
        votes[rows, winner] += 1
        bounded = margin / (1 + np.abs(margin))
        confidence[:, negative] -= bounded
        confidence[:, positive] += bounded
    # A correction smaller than 1/3 cannot overturn a one-vote advantage.
    scores = votes + confidence / (3 * (1 + np.abs(confidence)))
    return scores.argmax(axis=1)


class SVMClassifier:
    def __init__(self, mean, scale, pairs, models, metadata):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.models = models
        self.metadata = metadata
        self.kernel = metadata['kernel']
        self.gamma = metadata['gamma']
        k = len(metadata['classes'])
        expected = np.asarray(list(combinations(range(k), 2)))
        if (k < 2 or self.mean.ndim != 1 or len(self.mean) == 0 or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or not np.array_equal(self.pairs, expected)
                or len(models) != len(expected)):
            raise ValueError('Invalid SVM model dimensions or class pairs')
        kernel_matrix(np.zeros((1, len(self.mean))), np.zeros((1, len(self.mean))), self.kernel, self.gamma)
        for model in models:
            sv = model.support_vectors
            if (sv.ndim != 2 or sv.shape[1] != len(self.mean) or len(sv) == 0
                    or model.dual_coefficients.shape != (len(sv),)
                    or not np.isfinite(sv).all() or not np.isfinite(model.dual_coefficients).all()
                    or not np.isfinite(model.bias)):
                raise ValueError('Invalid binary SVM parameters')

    @classmethod
    def fit(cls, x, y, mean, scale, classes, c, kernel, gamma, solver_options, metadata):
        x = (x - mean) / scale
        if set(np.unique(y)) != set(range(len(classes))):
            raise ValueError('Training labels must cover each contiguous class index')
        pairs = list(combinations(range(len(classes)), 2))
        models = []
        for negative, positive in pairs:
            mask = (y == negative) | (y == positive)
            labels = np.where(y[mask] == positive, 1., -1.)
            try:
                models.append(fit_binary(x[mask], labels, c, kernel, gamma, solver_options))
            except (ValueError, ArithmeticError) as exc:
                raise ValueError(f'SVM pair {classes[negative]} / {classes[positive]}: {exc}') from exc
        metadata = {**metadata, 'version': 1, 'model_type': 'svm', 'classes': classes,
                    'C': c, 'kernel': kernel, 'gamma': gamma}
        return cls(mean, scale, pairs, models, metadata)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        standardized = (x - self.mean) / self.scale
        margins = np.column_stack([model.decision_function(standardized, self.kernel, self.gamma)
                                   for model in self.models])
        return vote_predictions(margins, self.pairs, len(self.metadata['classes']))

    def save(self, path):
        offsets = np.concatenate(([0], np.cumsum([len(m.support_vectors) for m in self.models])))
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, mean=self.mean, scale=self.scale, pairs=self.pairs,
                                offsets=offsets, support_vectors=np.vstack([m.support_vectors for m in self.models]),
                                dual_coefficients=np.concatenate([m.dual_coefficients for m in self.models]),
                                biases=np.asarray([m.bias for m in self.models]),
                                metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata.get('model_type') != 'svm' or metadata.get('version') != 1:
                raise ValueError('Invalid SVM model format')
            offsets, biases = archive['offsets'], archive['biases']
            sv, coef = archive['support_vectors'], archive['dual_coefficients']
            if (offsets.dtype.kind not in 'iu' or offsets.shape != (len(biases) + 1,)
                    or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
                    or offsets[-1] != len(sv) or coef.shape != (len(sv),)):
                raise ValueError('Invalid support-vector offsets')
            models = [BinarySVM(sv[a:b], coef[a:b], float(bias), {})
                      for a, b, bias in zip(offsets[:-1], offsets[1:], biases, strict=True)]
            return cls(archive['mean'], archive['scale'], archive['pairs'], models, metadata)
