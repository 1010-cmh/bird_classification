"""Device selection, reproducible state and CNN checkpoint serialization."""

import os
from pathlib import Path
import random

import numpy as np
import torch

from src.models.cnn import BirdCNN


PREPROCESSING = 'rgb_letterbox_bilinear_128_pixels_div_127.5_minus_1_v1'


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Device must be auto, cpu, cuda or mps')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    return torch.device(name)


def seed_everything(seed, device, cpu_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    # CPU/CUDA require deterministic operators; MPS reproducibility is checked empirically.
    torch.use_deterministic_algorithms(device.type != 'mps')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def capture_rng(device):
    np_state = np.random.get_state()
    result = {'python': random.getstate(),
              'numpy': [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
              'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state_all()
    elif device.type == 'mps':
        result['mps'] = torch.mps.get_rng_state()
    return cpu_copy(result)


def restore_rng(state, device):
    random.setstate(state['python'])
    kind, keys, position, has_gauss, cached = state['numpy']
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['mps'])


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1
            or checkpoint.get('model_type') != 'cnn'
            or checkpoint.get('preprocessing') != PREPROCESSING
            or checkpoint.get('kind') not in ('training', 'inference')):
        raise ValueError('Unsupported CNN checkpoint format')
    return checkpoint


def build_model(config, num_classes):
    return BirdCNN(num_classes, config['channels'], config['convolutions_per_stage'])


def load_inference_model(path, device):
    checkpoint = read_checkpoint(path)
    classes = checkpoint['classes']
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Invalid checkpoint classes')
    model = build_model(checkpoint['config'], len(classes))
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    return model, checkpoint
