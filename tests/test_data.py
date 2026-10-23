import copy
import shutil

import numpy as np
import pytest
import torch

from src.data import audit_dataset, load_feature, load_features, make_manifest, validate_manifest
from src.utils import digest


def test_stratified_manifest_is_reproducible_and_disjoint(dataset):
    summary, records = audit_dataset(dataset, expected_classes=3)
    assert not summary['errors']
    first = make_manifest(summary, records, num_classes=2)
    second = make_manifest(summary, records, num_classes=2)
    assert first == second
    assert len(first['selected_classes']) == 2
    assert set(first['class_to_idx'].values()) == {0, 1, 2}
    assert [len(first['splits'][s]) for s in ('train', 'internal_val', 'test')] == [30, 6, 9]
    assert all(row['source'] == 'train' for row in first['splits']['internal_val'])
    edited = copy.deepcopy(first)
    edited['selected_classes'].reverse()
    with pytest.raises(ValueError, match='checksum'):
        validate_manifest(edited)
    edited = copy.deepcopy(first)
    edited['splits']['internal_val'][0] = edited['splits']['train'][0]
    edited['manifest_sha256'] = digest({k: v for k, v in edited.items() if k != 'manifest_sha256'})
    with pytest.raises(ValueError, match='leaks'):
        validate_manifest(edited)


def test_duplicate_training_images_stay_in_same_split(dataset):
    folder = dataset / 'train/000.bird'
    shutil.copyfile(folder / '000.jpg', folder / '001.jpg')
    summary, records = audit_dataset(dataset, expected_classes=3)
    manifest = make_manifest(summary, records, num_classes=2)
    memberships = {r['image']: split for split, rows in manifest['splits'].items() for r in rows}
    assert memberships['train/000.bird/000.jpg'] == memberships['train/000.bird/001.jpg']


def test_cross_split_duplicates_block_preparation(dataset):
    shutil.copyfile(dataset / 'train/000.bird/000.jpg', dataset / 'val/000.bird/000.jpg')
    summary, records = audit_dataset(dataset, expected_classes=3)
    assert any('crosses original' in e for e in summary['errors'])
    with pytest.raises(ValueError, match='audit failed'):
        make_manifest(summary, records, num_classes=2)


def test_audit_reports_missing_corrupt_and_orphan_files(dataset):
    (dataset / 'train/000.bird/000.pt').unlink()
    (dataset / 'train/000.bird/001.jpg').write_bytes(b'not an image')
    torch.save(torch.ones(6), dataset / 'val/000.bird/orphan.pt')
    summary, _ = audit_dataset(dataset, expected_classes=3)
    assert len(summary['errors']) == 3


def test_feature_validation_and_explicit_dictionary_key(tmp_path):
    path = tmp_path / 'feature.pt'
    torch.save({'embedding': torch.ones(1, 8)}, path)
    vector, shape = load_feature(path, 'embedding')
    assert vector.shape == (8,) and shape == [1, 8]
    with pytest.raises(ValueError, match='Expected tensor'):
        load_feature(path)
    torch.save(torch.tensor([1., np.nan]), path)
    with pytest.raises(ValueError, match='finite feature'):
        load_feature(path)
    torch.save(torch.ones(2, 3), path)
    with pytest.raises(ValueError, match='feature vector'):
        load_feature(path)


def test_changed_feature_is_detected(dataset):
    summary, records = audit_dataset(dataset, expected_classes=3)
    manifest = make_manifest(summary, records, num_classes=3)
    row = manifest['splits']['train'][0]
    torch.save(torch.zeros(6), dataset / row['feature'])
    with pytest.raises(ValueError, match='changed since'):
        load_features(dataset, manifest, 'train')
