"""Audit course data and save a deterministic split manifest."""

import argparse
from pathlib import Path

from src.data import audit_dataset, make_manifest
from src.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('splits/cub200-seed42.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('outputs/audit.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--expected-classes', type=int, default=200)
    parser.add_argument('--feature-key')
    args = parser.parse_args()
    if args.output.exists() or args.audit_output.exists():
        parser.error('Output already exists; use a new output path to preserve the frozen split')
    if args.output.resolve() == args.audit_output.resolve():
        parser.error('Manifest and audit output paths must differ')
    summary, records = audit_dataset(args.data_dir, args.expected_classes, args.feature_key)
    write_json(args.audit_output, summary)
    manifest = make_manifest(summary, records, args.seed, args.validation_fraction, args.num_classes)
    write_json(args.output, manifest)
    print(f"Audit: {summary['totals']}; feature dimension: {manifest['feature_dimension']}")
    print(f"Manifest: {args.output}; sha256: {manifest['manifest_sha256']}")


if __name__ == '__main__':
    main()
