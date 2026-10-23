"""Multiclass linear classification with analytic gradients."""

import json
from pathlib import Path

import numpy as np


def fit_standardizer(x):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def loss_and_grad(x, y, weight, bias, l2=0.0):
    logits = x @ weight + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    sums = exp.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(sums[:, 0]) - shifted[np.arange(len(y)), y])
    loss += 0.5 * l2 * np.sum(weight * weight)
    residual = exp / sums
    residual[np.arange(len(y)), y] -= 1
    residual /= len(y)
    return float(loss), x.T @ residual + l2 * weight, residual.sum(axis=0)


class SoftmaxClassifier:
    def __init__(self, mean, scale, weight, bias, metadata):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.metadata = metadata
        d, k = self.weight.shape
        if (self.mean.shape != (d,) or self.scale.shape != (d,)
                or self.bias.shape != (k,) or len(metadata["classes"]) != k
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid model dimensions or standardization parameters")
        if not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.weight, self.bias)):
            raise ValueError("Non-finite model parameters")

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        return (((x - self.mean) / self.scale) @ self.weight + self.bias).argmax(axis=1)

    def save(self, path):
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, mean=self.mean, scale=self.scale,
                                weight=self.weight, bias=self.bias,
                                metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive["mean"], archive["scale"], archive["weight"], archive["bias"],
                       json.loads(str(archive["metadata"].item())))
