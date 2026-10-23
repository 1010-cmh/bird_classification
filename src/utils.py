"""Serialization, provenance and evaluation utilities."""

import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path outside dataset: {relative}")
    return path


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "Pillow", "torch")},
            "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "git_diff_sha256": digest(git("diff", "HEAD")),
            "source_sha256": {p.relative_to(repo).as_posix(): file_hash(p)
                              for p in sorted((repo / 'src').rglob('*.py'))}}


def classification_metrics(y, prediction, class_count):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.size == 0 or y.shape != prediction.shape:
        raise ValueError("Labels and predictions must have the same nonempty shape")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    correct = int(np.count_nonzero(y == prediction))
    return {"accuracy": correct / len(y), "correct": correct, "total": len(y),
            "confusion_matrix": matrix.tolist()}


def write_predictions(path, rows, y, prediction, classes):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "true_label", "predicted_label", "true_class", "predicted_class"])
        for row, truth, pred in zip(rows, y, prediction, strict=True):
            writer.writerow([row["image"], int(truth), int(pred), classes[truth], classes[pred]])
