#!/usr/bin/env python3
"""
Calculate plateau/stabilizing values from training results.
Reads results.txt files and computes moving averages and plateau values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from typing import Dict, Any

def calculate_plateau_values(results_root: Path) -> list[Dict[str, Any]]:
    """
    Calculate plateau values for all runs in results directory.
    
    Returns:
        List of dictionaries with batch_size, plateau values, and other metrics
    """
    results = []
    
    for folder in sorted(results_root.glob('*/')):
        results_file = folder / 'results.txt'
        if not results_file.exists():
            continue
        
        # Parse batch size and learning rate from folder name
        parts = folder.name.split('_')
        batch_size = None
        lr = None
        
        for part in parts:
            if part.startswith('b') and part[1:].isdigit():
                batch_size = int(part[1:])
            elif part.startswith('lr'):
                try:
                    lr = float(part[2:])
                except ValueError:
                    pass
        
        if batch_size is None:
            continue
        
        try:
            # Load data
            df = pd.read_csv(
                results_file,
                skiprows=4,
                sep=',',
                header=None,
                names=['epoch', 'step', 'batch_loss', 'full_loss', 'lambda_max',
                       'step_sharpness', 'batch_sharpness', 'gni', 'total_accuracy'],
                na_values=['nan'],
                skipinitialspace=True
            )
            
            if len(df) == 0:
                continue
            
            # Calculate moving averages (10% window)
            window = max(1, int(len(df) * 0.1))
            
            # Plateau: average of last 20% of steps
            plateau_start_idx = int(len(df) * 0.8)
            
            result = {
                'batch_size': batch_size,
                'learning_rate': lr if lr else None,
                'folder': folder.name,
                'total_steps': len(df),
                'final_step': int(df['step'].iloc[-1]) if 'step' in df.columns else None,
            }
            
            # Calculate plateau values for each metric
            for metric in ['batch_sharpness', 'lambda_max', 'step_sharpness', 'gni', 'full_loss']:
                if metric in df.columns:
                    series = df[metric].dropna()
                    if len(series) > 0:
                        # Moving average
                        ma = series.rolling(window=window, center=True, min_periods=1).mean()
                        result[f'{metric}_moving_avg_final'] = float(ma.iloc[-1]) if len(ma) > 0 else None
                        
                        # Plateau value (last 20% average)
                        plateau_series = series.iloc[plateau_start_idx:] if plateau_start_idx < len(series) else series
                        result[f'{metric}_plateau'] = float(plateau_series.mean()) if len(plateau_series) > 0 else None
                        result[f'{metric}_plateau_std'] = float(plateau_series.std()) if len(plateau_series) > 0 else None
                        
                        # Final value
                        result[f'{metric}_final'] = float(series.iloc[-1]) if len(series) > 0 else None
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {folder.name}: {e}", file=os.sys.stderr)
            continue
    
    return sorted(results, key=lambda x: x['batch_size'])


def main():
    results_root = Path(os.environ.get('RESULTS', '~/results')) / 'plaintext' / 'cifar10_mlp'
    
    if not results_root.exists():
        print(f"Error: Results directory not found: {results_root}")
        print("Set RESULTS environment variable or ensure ~/results/plaintext/cifar10_mlp exists")
        return 1
    
    print(f"Analyzing results in: {results_root}")
    print("Calculating plateau values...")
    
    results = calculate_plateau_values(results_root)
    
    if not results:
        print("No results found!")
        return 1
    
    # Save as JSON
    output_json = Path(__file__).parent / 'plateau_values.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON results to: {output_json}")
    
    # Save as CSV (flattened for easier viewing)
    output_csv = Path(__file__).parent / 'plateau_values.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV results to: {output_csv}")
    
    # Print summary
    print(f"\nFound {len(results)} completed runs")
    print("\nSummary (batch_size, batch_sharpness_plateau, lambda_max_plateau):")
    for r in results[:10]:  # Show first 10
        bs = r.get('batch_size', 'N/A')
        bsp = r.get('batch_sharpness_plateau', None)
        lmp = r.get('lambda_max_plateau', None)
        print(f"  Batch {bs:4d}: batch_sharpness={bsp:.4f if bsp else 'N/A':>8}, lambda_max={lmp:.4f if lmp else 'N/A':>8}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

