"""Dataset auditing, frozen splits and feature loading."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils import dataset_path, digest, file_hash, read_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_feature(path, key=None):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing feature key {key}: {path}")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected tensor; use --feature-key for a tensor in a dictionary: {path}")
    if value.is_complex() or not value.is_floating_point():
        raise ValueError(f"Expected real floating-point feature: {path}")
    shape = list(value.shape)
    value = value.detach().to(dtype=torch.float64).numpy().squeeze()
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"Expected a finite feature vector, got shape {shape}: {path}")
    return value, shape


def audit_dataset(root, expected_classes=200, feature_key=None):
    root = Path(root).resolve()
    errors, records = [], []
    counts, dimensions, modes, shapes = {}, Counter(), Counter(), Counter()
    feature_min, feature_max = None, None
    class_lists = {}
    for split in ("train", "val"):
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f"Missing dataset directory: {directory}")
        classes = sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
        class_lists[split] = classes
        counts[split] = {}
        for name in classes:
            folder = directory / name
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            counts[split][name] = len(images)
            if not images:
                errors.append(f"Empty class: {split}/{name}")
            paired = {p.with_suffix('.pt') for p in images}
            for orphan in sorted(set(folder.glob('*.pt')) - paired):
                errors.append(f"Feature without image: {orphan.relative_to(root)}")
            for path in images:
                relative = path.relative_to(root).as_posix()
                feature = path.with_suffix(".pt")
                try:
                    with Image.open(path) as im:
                        im.load()
                        width, height, mode = im.width, im.height, im.mode
                    vector, shape = load_feature(feature, feature_key)
                    dimensions[len(vector)] += 1
                    lo, hi = float(vector.min()), float(vector.max())
                    feature_min = lo if feature_min is None else min(lo, feature_min)
                    feature_max = hi if feature_max is None else max(hi, feature_max)
                    modes[mode] += 1
                    shapes[str(shape)] += 1
                    records.append({"image": relative,
                                    "feature": feature.relative_to(root).as_posix(),
                                    "class": name, "source": split,
                                    "image_sha256": file_hash(path),
                                    "feature_sha256": file_hash(feature),
                                    "width": width, "height": height})
                except Exception as exc:
                    errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    if class_lists["train"] != class_lists["val"]:
        errors.append("Training and test class directories do not match")
    if len(class_lists["train"]) != expected_classes:
        errors.append(f"Expected {expected_classes} classes; found {len(class_lists['train'])}")
    if len(dimensions) != 1:
        errors.append(f"Inconsistent or missing feature dimensions: {dict(dimensions)}")
    by_hash = defaultdict(list)
    for row in records:
        by_hash[row["image_sha256"]].append(row)
    duplicates = [[r["image"] for r in group] for group in by_hash.values() if len(group) > 1]
    for group in by_hash.values():
        if len({r['source'] for r in group}) > 1:
            errors.append("Duplicate image crosses original train/test split: " + str([r['image'] for r in group]))
        if len({r['class'] for r in group}) > 1:
            errors.append("Duplicate image has conflicting classes: " + str([r['image'] for r in group]))
    summary = {"classes": class_lists["train"], "counts": counts,
               "totals": {s: sum(c.values()) for s, c in counts.items()},
               "feature_dimensions": dict(dimensions), "feature_shapes": dict(shapes),
               "feature_min": feature_min, "feature_max": feature_max,
               "image_size_range": {
                   axis: [min(r[axis] for r in records), max(r[axis] for r in records)]
                   for axis in ('width', 'height')} if records else {},
               "image_modes": dict(modes), "duplicate_images": duplicates,
               "errors": errors, "feature_key": feature_key,
               "dataset_fingerprint": digest(records)}
    return summary, records


def make_manifest(summary, records, seed=42, validation_fraction=0.15, num_classes=10):
    if summary["errors"]:
        raise ValueError("Dataset audit failed; inspect the audit report")
    classes = summary["classes"]
    if not 0 < validation_fraction < 1 or not 2 <= num_classes <= len(classes):
        raise ValueError("Invalid validation fraction or selected class count")
    selected = sorted(np.random.default_rng(seed).choice(classes, num_classes, replace=False).tolist())
    rng = np.random.default_rng(seed)
    splits = {"train": [], "internal_val": [], "test": []}
    for name in classes:
        train = [r for r in records if r["class"] == name and r["source"] == "train"]
        groups = defaultdict(list)
        for row in train:
            groups[row["image_sha256"]].append(row)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError(f"At least two distinct training images required: {name}")
        order = rng.permutation(len(keys))
        target = max(1, min(len(train) - 1, int(np.floor(len(train) * validation_fraction + 0.5))))
        validation_hashes, count = set(), 0
        for i in order[:-1]:
            if count >= target:
                break
            validation_hashes.add(keys[i])
            count += len(groups[keys[i]])
        for row in train:
            split = "internal_val" if row["image_sha256"] in validation_hashes else "train"
            splits[split].append(row)
        splits["test"].extend(r for r in records if r["class"] == name and r["source"] == "val")
    for rows in splits.values():
        rows.sort(key=lambda row: row["image"])
    manifest = {"version": 1, "seed": seed, "validation_fraction": validation_fraction,
                "class_to_idx": {name: i for i, name in enumerate(classes)},
                "selected_classes": selected, "feature_key": summary["feature_key"],
                "feature_dimension": int(next(iter(summary["feature_dimensions"]))),
                "dataset_fingerprint": summary["dataset_fingerprint"], "splits": splits}
    manifest["manifest_sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("version") != 1 or digest(payload) != manifest.get("manifest_sha256"):
        raise ValueError("Invalid manifest version or checksum")
    mapping = manifest["class_to_idx"]
    classes = sorted(mapping)
    if mapping != {name: i for i, name in enumerate(classes)}:
        raise ValueError("Invalid class mapping")
    selected = manifest["selected_classes"]
    if len(selected) < 2 or len(set(selected)) != len(selected) or not set(selected) <= set(classes):
        raise ValueError("Invalid selected classes")
    images, features, hashes = set(), set(), set()
    for split in ("train", "internal_val", "test"):
        rows = manifest["splits"][split]
        if {r["class"] for r in rows} != set(classes):
            raise ValueError(f"Missing or unknown classes in {split}")
        split_hashes = set()
        source = "val" if split == "test" else "train"
        for row in rows:
            if row["source"] != source:
                raise ValueError(f"Incorrect source split: {row['image']}")
            for field in ("image", "feature"):
                p = Path(row[field])
                if p.is_absolute() or len(p.parts) != 3 or p.parts[:2] != (source, row["class"]):
                    raise ValueError(f"Invalid manifest path: {p}")
            if row['image'] in images or row['feature'] in features or row['image_sha256'] in hashes:
                raise ValueError("Sample or duplicate image leaks across splits")
            images.add(row['image'])
            features.add(row['feature'])
            split_hashes.add(row['image_sha256'])
        hashes.update(split_hashes)
    records = sorted((r for rows in manifest['splits'].values() for r in rows),
                     key=lambda r: (r['source'], r['class'], r['image']))
    if digest(records) != manifest['dataset_fingerprint']:
        raise ValueError("Dataset fingerprint does not match manifest records")


def load_manifest(path):
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def load_features(root, manifest, split):
    classes = manifest["selected_classes"]
    labels = {name: i for i, name in enumerate(classes)}
    rows = [r for r in manifest["splits"][split] if r["class"] in labels]
    vectors = []
    for row in rows:
        path = dataset_path(root, row["feature"])
        if file_hash(path) != row["feature_sha256"]:
            raise ValueError(f"Feature changed since dataset audit: {row['feature']}")
        vector, _ = load_feature(path, manifest['feature_key'])
        if len(vector) != manifest['feature_dimension']:
            raise ValueError(f"Feature dimension mismatch: {path}")
        vectors.append(vector)
    return np.stack(vectors), np.asarray([labels[r['class']] for r in rows]), rows
