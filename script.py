import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_heartbeats(record_id, window_size=180):
    # Construct paths based on your layout
    ekg_path = f'raw_data/{record_id}_ekg.csv'
    ann_path = f'raw_data/{record_id}_annotations_1.csv'

    # Load the signal (Voltage) and the markers (R-peak indices)
    df_signal = pd.read_csv(ekg_path)
    df_ann = pd.read_csv(ann_path)

    # MLII is the standard lead for arrhythmia detection
    signal = df_signal['MLII'].values

    # Updated to match your specific CSV columns:
    peak_indices = df_ann['index'].values
    labels = df_ann['annotation_symbol'].values

    beats = []
    beat_labels = []

    half_win = window_size // 2

    for i in range(len(peak_indices)):
        p = peak_indices[i]
        label = labels[i]

        # Ensure we have enough data points before and after the peak
        if p > half_win and p < len(signal) - half_win:
            segment = signal[p - half_win: p + half_win]

            # Hardware-friendly Normalization (0 to 1)
            norm_segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))

            beats.append(norm_segment)
            beat_labels.append(label)

    return np.array(beats), beat_labels


# List of records to include
record_ids = ['100', '106', '119']

all_beats = []
all_labels = []

for rid in record_ids:
    print(f"Processing record {rid}...")
    beats, labels = extract_heartbeats(rid)
    all_beats.append(beats)
    all_labels.extend(labels)

# Combine everything into large arrays
X = np.vstack(all_beats)
y = np.array(all_labels)

print(f"Final dataset shape: {X.shape}")
print(f"Label counts: {pd.Series(y).value_counts()}")
# 1. Create a mask to keep only 'N' and 'V' beats
mask = np.isin(y, ['N', 'V'])
X_filtered = X[mask]
y_filtered = y[mask]

# 2. Convert text labels to binary (0 for Normal, 1 for Ventricular)
# For H/W 0/1 is much easier to handle than ASCII 'N'/'V'
y_binary = np.where(y_filtered == 'N', 0, 1)

print("\n--- Filtered Dataset ---")
print(f"Remaining beats (N and V only): {len(y_binary)}")
print(f"Normal (0) count: {np.sum(y_binary == 0)}")
print(f"Arrhythmia (1) count: {np.sum(y_binary == 1)}")

# 3. Save the clean data for the next step-training
np.save('X_final.npy', X_filtered)
np.save('y_final.npy', y_binary)
print("\nFiles saved: X_final.npy and y_final.npy")