import copy
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cnn_runtime import load_inference_model, read_checkpoint
from src.data import audit_dataset, make_manifest
from src.evaluate_cnn import evaluate
from src.image_data import BirdImageDataset, image_tensor
from src.models.cnn import BirdCNN
from src.predict import predict
from src.train_cnn import run_epoch, train, validate_config
from src.utils import write_json


CONFIG = {'seed': 7, 'image_size': 16, 'channels': [4, 8], 'convolutions_per_stage': 1,
          'batch_size': 6, 'epochs': 3, 'learning_rate': 0.05, 'momentum': 0.9,
          'weight_decay': 0.0001, 'num_workers': 0, 'cpu_threads': 1}


def make_split(dataset, tmp_path):
    report, records = audit_dataset(dataset, expected_classes=3)
    manifest = make_manifest(report, records, num_classes=2)
    path = tmp_path / 'manifest.json'
    write_json(path, manifest)
    return manifest, path


def test_preprocessing_handles_grayscale_and_keeps_aspect_ratio():
    tensor = image_tensor(Image.new('L', (20, 10), color=255), 16)
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[1], tensor[2])
    assert torch.all(tensor[:, 4:12] == 1)
    np.testing.assert_allclose(tensor[:, :4], 128 / 127.5 - 1, atol=1e-7)


def test_image_dataset_uses_all_classes_and_never_features(dataset, tmp_path):
    manifest, _ = make_split(dataset, tmp_path)
    for path in dataset.rglob('*.pt'):
        path.unlink()
    shutil.rmtree(dataset / 'val')
    images = BirdImageDataset(dataset, manifest, 'train', 16)
    assert len(images.classes) == 3 and len(manifest['selected_classes']) == 2
    assert {images[i][1] for i in range(len(images))} == {0, 1, 2}
    assert torch.equal(images[0][0], images[0][0])
    images.paths[0].write_bytes(b'changed image')
    with pytest.raises(ValueError, match='changed since'):
        BirdImageDataset(dataset, manifest, 'train', 16)


def test_network_can_overfit_small_training_batch():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    images = torch.zeros(12, 3, 16, 16)
    labels = torch.arange(3).repeat_interleave(4)
    for i, label in enumerate(labels):
        images[i, label] = 1
    model = BirdCNN(3, [8, 16], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = DataLoader(TensorDataset(images, labels), batch_size=12)
    first, _, _ = run_epoch(model, loader, torch.device('cpu'), optimizer)
    for _ in range(39):
        last, _, _ = run_epoch(model, loader, torch.device('cpu'), optimizer)
    evaluated, _, _ = run_epoch(model, loader, torch.device('cpu'))
    assert evaluated['accuracy'] == 1
    assert last['loss'] < first['loss'] / 10
    assert model(images).shape == (12, 3)


def test_resume_matches_uninterrupted_training_and_preserves_best_model(dataset, tmp_path):
    manifest, manifest_path = make_split(dataset, tmp_path)
    for path in dataset.rglob('*.pt'):
        path.unlink()
    shutil.rmtree(dataset / 'val')
    continuous = train(dataset, manifest_path, CONFIG, tmp_path / 'continuous', 'cpu')
    first = train(dataset, manifest_path, CONFIG, tmp_path / 'first', 'cpu', stop_after_epoch=1)
    assert first['schedule_complete'] is False
    continued = train(dataset, manifest_path, CONFIG, tmp_path / 'continued', 'cpu',
                      resume=tmp_path / 'first/last.pt')
    left, right = read_checkpoint(tmp_path / 'continuous/last.pt'), read_checkpoint(tmp_path / 'continued/last.pt')
    for key in left['model_state']:
        assert torch.equal(left['model_state'][key], right['model_state'][key]), key
    for key in left['best_model_state']:
        assert torch.equal(left['best_model_state'][key], right['best_model_state'][key]), key
    assert left['scheduler_state'] == right['scheduler_state']
    assert torch.equal(left['train_loader_rng'], right['train_loader_rng'])
    assert torch.equal(left['rng_state']['torch'], right['rng_state']['torch'])
    assert continuous['best_epoch'] == continued['best_epoch']
    assert continued['num_classes'] == 3 and continued['test_evaluated'] is False
    metrics = evaluate(dataset, manifest_path, tmp_path / 'continued/model.pt', 'internal_val',
                       tmp_path / 'evaluation', 'cpu')
    assert metrics['accuracy'] == continued['internal_val']['accuracy']
    assert metrics['loss'] == pytest.approx(continued['internal_val']['loss'], abs=1e-7)
    image = dataset / manifest['splits']['internal_val'][0]['image']
    result = predict(image, tmp_path / 'continued/model.pt', 'cpu', top_k=3)
    assert len(result['predictions']) == 3
    assert sum(p['probability'] for p in result['predictions']) == pytest.approx(1, abs=1e-6)
    restored, _ = load_inference_model(tmp_path / 'continued/model.pt', torch.device('cpu'))
    with Image.open(image) as im:
        pixels = image_tensor(im, CONFIG['image_size']).unsqueeze(0)
    with torch.inference_mode():
        expected = int(restored(pixels).argmax())
    assert result['predictions'][0]['label'] == expected
    with pytest.raises(ValueError, match='same training config'):
        train(dataset, manifest_path, {**CONFIG, 'epochs': 4}, tmp_path / 'invalid', 'cpu',
              resume=tmp_path / 'first/last.pt')
    with pytest.raises(ValueError, match='same training config'):
        train(dataset, manifest_path, CONFIG, tmp_path / 'inference-resume', 'cpu',
              resume=tmp_path / 'first/model.pt')


def test_evaluation_does_not_update_batchnorm_or_weights():
    torch.manual_seed(1)
    model = BirdCNN(3, [4], 1)
    before = copy.deepcopy(model.state_dict())
    loader = DataLoader(TensorDataset(torch.randn(5, 3, 16, 16), torch.tensor([0, 1, 2, 0, 1])), batch_size=3)
    metrics, labels, predictions = run_epoch(model, loader, torch.device('cpu'))
    assert metrics['total'] == 5 and len(labels) == len(predictions) == 5
    for key in before:
        assert torch.equal(before[key], model.state_dict()[key])


@pytest.mark.parametrize('update', [{'epochs': 0}, {'image_size': 8}, {'num_workers': -1},
                                    {'channels': []}, {'momentum': 1}, {'learning_rate': float('nan')},
                                    {'weight_decay': -1}])
def test_invalid_cnn_configuration(update):
    with pytest.raises(ValueError):
        validate_config({**CONFIG, **update})
