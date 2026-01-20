#!/usr/bin/env python3
"""
Add datapoints to plateau values dataframe with proper learning_rate.
"""

import pandas as pd

# Load your data
pv001 = pd.read_csv("stability_mlp_lr001_mom9_N8192.csv").iloc[:-1]
pv003 = pv001[pv001["learning_rate"] == 0.001]

print(f"Original rows after filtering: {len(pv003)}")

# Create new rows WITH learning_rate set to 0.001
new_rows = pd.DataFrame({
    'batch_size': [92, 1090, 8192],
    'learning_rate': [0.001, 0.001, 0.001],  # Set learning_rate to match filter
    'batch_sharpness_plateau': [550, 3470, 3680],
    'lambda_max_plateau': [775, 3750, 3800]
})

# Add to existing dataframe
pv003 = pd.concat([pv003, new_rows], ignore_index=True)

print(f"After adding rows: {len(pv003)}")

# Sort by batch_size
pv003 = pv003.sort_values('batch_size').reset_index(drop=True)

# Verify the new rows are there
print(f"\nLast 5 rows:")
print(pv003.tail())

# Save if needed
# pv003.to_csv('output.csv', index=False)
