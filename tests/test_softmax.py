import numpy as np
import pytest

from src.traditional.softmax import SoftmaxClassifier, fit_standardizer, loss_and_grad


def test_analytic_gradients_match_finite_differences():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4))
    y = np.array([0, 2, 1, 2, 0])
    weight = rng.normal(size=(4, 3))
    bias = rng.normal(size=3)
    _, dw, db = loss_and_grad(x, y, weight, bias, l2=0.2)
    for parameter, analytic in ((weight, dw), (bias, db)):
        numeric = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = loss_and_grad(x, y, weight, bias, l2=0.2)[0]
            parameter[index] = original - 1e-6
            minus = loss_and_grad(x, y, weight, bias, l2=0.2)[0]
            parameter[index] = original
            numeric[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)


def test_large_logits_have_finite_loss_and_gradients():
    result = loss_and_grad(np.array([[1000.0], [-1000.0]]), np.array([1, 0]),
                           np.array([[1000.0, -1000.0]]), np.zeros(2))
    assert all(np.isfinite(value).all() for value in result)
    assert result[0] == pytest.approx(2e6)


def test_constant_feature_and_model_round_trip(tmp_path):
    x = np.array([[1., 5.], [3., 5.], [5., 5.]])
    mean, scale = fit_standardizer(x)
    assert scale[1] == 1
    model = SoftmaxClassifier(mean, scale, np.array([[1., -1.], [0., 0.]]),
                              np.zeros(2), {'classes': ['a', 'b']})
    path = tmp_path / 'model.npz'
    model.save(path)
    restored = SoftmaxClassifier.load(path)
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
    with pytest.raises(ValueError, match='Invalid feature'):
        restored.predict(np.array([[np.nan, 5.]]))
