#!/usr/bin/env python3
"""
Convert results.txt files to CSV with step, batch_sharpness, and lmax columns.

Usage:
    # Convert a single file:
    python convert_results_to_csv.py /path/to/results.txt

    # Convert multiple files:
    python convert_results_to_csv.py /path/to/dir1/results.txt /path/to/dir2/results.txt

    # Convert all results.txt in a parent directory:
    python convert_results_to_csv.py --dir /path/to/cifar10_mlp --pattern "2026*"

Output files are saved next to source or in --output-dir if specified.
"""

import argparse
import csv
import re
from pathlib import Path


def parse_results_txt(filepath: Path) -> list[dict]:
    """Parse results.txt and extract step, batch_sharpness, lmax."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue

            # Parse: epoch, step, batch_loss, full_loss, lmax, step_sharpness, batch_sharpness, gni, accuracy
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue

            try:
                step = int(parts[1])
                lmax = parts[4].strip()  # column 4 is lambda_max
                batch_sharpness = parts[6].strip()  # column 6 is batch_sharpness

                # Convert 'nan' to empty or keep as-is for CSV
                lmax_val = float(lmax) if lmax.lower() != 'nan' else None
                bs_val = float(batch_sharpness) if batch_sharpness.lower() != 'nan' else None

                rows.append({
                    'step': step,
                    'lmax': lmax_val,
                    'batch_sharpness': bs_val,
                })
            except (ValueError, IndexError) as e:
                continue

    return rows


def extract_lr_from_path(filepath: Path) -> str:
    """Extract learning rate from directory name like 20260128_1436_25_lr0.02000_b8."""
    dir_name = filepath.parent.name
    match = re.search(r'lr([\d.]+)', dir_name)
    if match:
        return match.group(1)
    return "unknown"


def convert_to_csv(input_path: Path, output_path: Path):
    """Convert a results.txt to CSV."""
    rows = parse_results_txt(input_path)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'lmax', 'batch_sharpness'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Converted {input_path} -> {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Convert results.txt to CSV')
    parser.add_argument('files', nargs='*', help='results.txt files to convert')
    parser.add_argument('--dir', type=str, help='Parent directory containing experiment folders')
    parser.add_argument('--pattern', type=str, default='*', help='Glob pattern for subdirectories (default: *)')
    parser.add_argument('--output-dir', type=str, help='Output directory for CSV files (default: same as input)')
    args = parser.parse_args()

    input_files = []

    # Collect files from positional arguments
    for f in args.files:
        p = Path(f)
        if p.is_file():
            input_files.append(p)
        elif p.is_dir():
            # If directory given, look for results.txt inside
            results_file = p / 'results.txt'
            if results_file.exists():
                input_files.append(results_file)

    # Collect files from --dir with pattern
    if args.dir:
        base_dir = Path(args.dir)
        for subdir in base_dir.glob(args.pattern):
            if subdir.is_dir():
                results_file = subdir / 'results.txt'
                if results_file.exists():
                    input_files.append(results_file)

    if not input_files:
        print("No results.txt files found. Provide files as arguments or use --dir")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in sorted(input_files):
        lr = extract_lr_from_path(input_file)

        if output_dir:
            output_name = f"{input_file.parent.name}.csv"
            output_path = output_dir / output_name
        else:
            output_path = input_file.parent / f"results_lr{lr}.csv"

        convert_to_csv(input_file, output_path)


if __name__ == '__main__':
    main()
