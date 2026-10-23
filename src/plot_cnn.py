"""Plot CNN loss and accuracy from an existing training history."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(history_path, output_path):
    with Path(history_path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty training history')
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for prefix, label, color in [('train', 'Training', '#2563eb'), ('internal_val', 'Internal validation', '#ea580c')]:
        axes[0].plot(epochs, [float(row[f'{prefix}_loss']) for row in rows], label=label, color=color)
        axes[1].plot(epochs, [100 * float(row[f'{prefix}_accuracy']) for row in rows], label=label, color=color)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Cross-entropy')
    axes[0].set_title('Loss')
    axes[1].set_ylabel('Top-1 accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim(0, 100)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.history, args.output)


if __name__ == '__main__':
    main()
