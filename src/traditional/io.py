"""Load saved classifiers, including Softmax checkpoints from the first experiment."""

import json

import numpy as np

from src.traditional.softmax import SoftmaxClassifier


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
    model_type = metadata.get('model_type', 'softmax')
    if model_type == 'softmax':
        return SoftmaxClassifier.load(path)
    if model_type == 'svm':
        from src.traditional.svm import SVMClassifier
        return SVMClassifier.load(path)
    raise ValueError(f'Unknown model type: {model_type}')
