"""Train a randomly initialized CNN on all classes with epoch-boundary resume."""

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn_runtime import (PREPROCESSING, build_model, capture_rng, cpu_copy, read_checkpoint,
                             resolve_device, restore_rng, save_checkpoint, seed_everything, synchronize)
from src.data import load_manifest
from src.image_data import BirdImageDataset
from src.utils import environment, file_hash, new_output, read_json, write_json


def validate_config(config):
    keys = {'seed', 'image_size', 'channels', 'convolutions_per_stage', 'batch_size', 'epochs',
            'learning_rate', 'momentum', 'weight_decay', 'num_workers', 'cpu_threads'}
    if set(config) != keys:
        raise ValueError(f'CNN configuration keys must be {sorted(keys)}')
    for key in ('seed', 'image_size', 'convolutions_per_stage', 'batch_size', 'epochs', 'num_workers', 'cpu_threads'):
        minimum = 0 if key in ('seed', 'num_workers') else 1
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'Invalid {key}')
    if config['seed'] >= 2**32 or config['image_size'] < 16:
        raise ValueError('Seed must fit uint32 and image size must be at least 16')
    channels = config['channels']
    if (not isinstance(channels, list) or not channels
            or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in channels)):
        raise ValueError('Channels must be a nonempty list of positive integers')
    if 2**len(channels) > config['image_size']:
        raise ValueError('Too many downsampling stages for image size')
    for key in ('learning_rate', 'momentum', 'weight_decay'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'Invalid {key}')
    if config['learning_rate'] <= 0 or not 0 <= config['momentum'] < 1 or config['weight_decay'] < 0:
        raise ValueError('Invalid optimizer parameters')


def run_epoch(model, loader, device, optimizer=None, progress=False):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, count = 0., 0, 0
    predictions, labels = [], []
    synchronize(device)
    started = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError('Non-finite CNN loss')
            if training:
                loss.backward()
                optimizer.step()
            prediction = logits.argmax(dim=1)
            batch_size = len(targets)
            loss_sum += float(loss.item()) * batch_size
            correct += int((prediction == targets).sum().item())
            count += batch_size
            if not training:
                predictions.extend(prediction.cpu().tolist())
                labels.extend(targets.cpu().tolist())
            if progress and batch % 50 == 0:
                print(f'  batch {batch}/{len(loader)}: loss={loss_sum/count:.4f}, accuracy={correct/count:.4f}', flush=True)
    if count == 0:
        raise ValueError('Empty image loader')
    synchronize(device)
    return {'loss': loss_sum / count, 'accuracy': correct / count, 'correct': correct,
            'total': count, 'seconds': time.perf_counter() - started}, labels, predictions


def write_history(path, history):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def inference_checkpoint(base, state, epoch, metrics):
    return {**base, 'kind': 'inference', 'model_state': state, 'epoch': epoch, 'selection': metrics}


def train(data_dir, manifest_path, config, output_dir, device_name='auto', resume=None, stop_after_epoch=None):
    validate_config(config)
    device = resolve_device(device_name)
    manifest = load_manifest(manifest_path)
    classes = sorted(manifest['class_to_idx'], key=manifest['class_to_idx'].get)
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint:
        if (checkpoint['kind'] != 'training' or checkpoint['config'] != config
                or checkpoint['manifest_sha256'] != manifest['manifest_sha256']
                or checkpoint['classes'] != classes or checkpoint['device_type'] != device.type):
            raise ValueError('Resume requires the same training config, manifest, classes and device type')
    start_epoch = checkpoint['epoch'] if checkpoint else 0
    end_epoch = config['epochs'] if stop_after_epoch is None else stop_after_epoch
    if not isinstance(end_epoch, int) or not start_epoch < end_epoch <= config['epochs']:
        raise ValueError('Stop epoch must be after the saved epoch and within the configured schedule')
    seed_everything(config['seed'], device, config['cpu_threads'])
    train_data = BirdImageDataset(data_dir, manifest, 'train', config['image_size'])
    val_data = BirdImageDataset(data_dir, manifest, 'internal_val', config['image_size'])
    output = new_output(output_dir)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    provenance = environment()
    provenance.update({'device': str(device), 'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else str(device),
                       'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'resume_checkpoint_sha256': file_hash(resume) if resume else None,
                       'starting_epoch': start_epoch, 'stop_after_epoch': end_epoch})
    write_json(output / 'environment.json', provenance)
    train_generator = torch.Generator().manual_seed(config['seed'])
    val_generator = torch.Generator().manual_seed(config['seed'] + 1)
    loader_options = {'batch_size': config['batch_size'], 'num_workers': config['num_workers'],
                      'pin_memory': device.type == 'cuda', 'persistent_workers': False}
    train_loader = DataLoader(train_data, shuffle=True, generator=train_generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, generator=val_generator, **loader_options)
    model = build_model(config, len(classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    history, best_state, best_metrics, best_epoch = [], None, None, 0
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        train_generator.set_state(checkpoint['train_loader_rng'])
        val_generator.set_state(checkpoint['val_loader_rng'])
        history, best_state = checkpoint['history'], checkpoint['best_model_state']
        best_metrics, best_epoch = checkpoint['best_metrics'], checkpoint['best_epoch']
        restore_rng(checkpoint['rng_state'], device)
    base = {'format_version': 1, 'model_type': 'cnn', 'config': config,
            'preprocessing': PREPROCESSING, 'classes': classes,
            'manifest_sha256': manifest['manifest_sha256'], 'device_type': device.type}
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}; classes: {len(classes)}; parameters: {parameter_count}; '
          f'train: {len(train_data)}; internal_val: {len(val_data)}', flush=True)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        learning_rate = optimizer.param_groups[0]['lr']
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, progress=True)
        val_metrics, _, _ = run_epoch(model, val_loader, device)
        scheduler.step()
        row = {'epoch': epoch, 'learning_rate': learning_rate, 'train_loss': train_metrics['loss'],
               'train_accuracy': train_metrics['accuracy'], 'internal_val_loss': val_metrics['loss'],
               'internal_val_accuracy': val_metrics['accuracy'], 'internal_val_correct': val_metrics['correct'],
               'train_seconds': train_metrics['seconds'], 'internal_val_seconds': val_metrics['seconds']}
        history.append(row)
        if best_metrics is None or (val_metrics['accuracy'], -val_metrics['loss']) > (best_metrics['accuracy'], -best_metrics['loss']):
            best_state, best_metrics, best_epoch = cpu_copy(model.state_dict()), val_metrics, epoch
        full = {**base, 'kind': 'training', 'epoch': epoch, 'model_state': cpu_copy(model.state_dict()),
                'optimizer_state': cpu_copy(optimizer.state_dict()), 'scheduler_state': scheduler.state_dict(),
                'rng_state': capture_rng(device), 'train_loader_rng': train_generator.get_state(),
                'val_loader_rng': val_generator.get_state(), 'history': history,
                'best_model_state': best_state, 'best_metrics': best_metrics, 'best_epoch': best_epoch}
        save_checkpoint(output / 'last.pt', full)
        save_checkpoint(output / 'model.pt', inference_checkpoint(base, best_state, best_epoch, best_metrics))
        write_history(output / 'history.csv', history)
        summary = {'task': 'image_classification', 'num_classes': len(classes), 'classes': classes,
                   'manifest_sha256': manifest['manifest_sha256'], 'parameter_count': parameter_count,
                   'fit_samples': len(train_data), 'internal_val_samples': len(val_data),
                   'epochs_completed': epoch, 'epochs_planned': config['epochs'],
                   'schedule_complete': epoch == config['epochs'], 'best_epoch': best_epoch,
                   'internal_val': best_metrics,
                   'selection_rule': 'highest internal validation accuracy; lowest cross-entropy breaks ties',
                   'initialization': 'random', 'test_evaluated': False, 'device': str(device)}
        write_json(output / 'summary.json', summary)
        print(f"Epoch {epoch}/{config['epochs']}: train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
              f"seconds={train_metrics['seconds'] + val_metrics['seconds']:.1f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/cnn-baseline.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    train(args.data_dir, args.manifest, read_json(args.config), args.output_dir,
          args.device, args.resume, args.stop_after_epoch)


if __name__ == '__main__':
    main()
