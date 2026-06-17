import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
DATA_DIR = 'raw_data'
WINDOW_SIZE = 180
def extract_heartbeats(record_id):
    ekg_path = os.path.join(DATA_DIR, f"{record_id}_ekg.csv")
    ann_path = os.path.join(DATA_DIR, f"{record_id}_annotations_1.csv")

    if not os.path.exists(ekg_path) or not os.path.exists(ann_path):
        return None, None

    df_signal = pd.read_csv(ekg_path)
    df_ann = pd.read_csv(ann_path)

    # --- COLUMN DETECTION ---
    if 'MLII' in df_signal.columns:
        signal = df_signal['MLII'].values
    else:
        # If MLII is missing, take the first column that isn't 'time'
        alt_col = [c for c in df_signal.columns if c.lower() != 'time'][0]
        print(f" Record {record_id}: 'MLII' not found. Using '{alt_col}' instead.   ")
        signal = df_signal[alt_col].values
    # ------------------------------------

    peak_indices = df_ann['index'].values
    labels = df_ann['annotation_symbol'].values

    beats, beat_labels = [], []
    half_win = WINDOW_SIZE // 2

    for i in range(len(peak_indices)):
        p = peak_indices[i]
        label = labels[i]

        if label in ['N', 'V'] and p > half_win and p < (len(signal) - half_win):
            segment = signal[p - half_win: p + half_win]

            denom = (np.max(segment) - np.min(segment))
            if denom == 0: continue
            norm_segment = (segment - np.min(segment)) / denom

            beats.append(norm_segment)
            beat_labels.append(0 if label == 'N' else 1)

    return np.array(beats), beat_labels

# --- 2. MAIN EXECUTION ---
# Auto-detect all IDs from /rawdata
record_ids = [f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith('_ekg.csv')]
record_ids.sort()

all_x, all_y = [], []

print(f"Starting processing of {len(record_ids)} records...")

# Process each file with a progress bar
for i, rid in enumerate(record_ids):
    print(f"Processing [{i + 1}/{len(record_ids)}]: Record {rid}...", end='\r')

    bx, by = extract_heartbeats(rid)
    if bx is not None and len(bx) > 0:
        all_x.append(bx)
        all_y.extend(by)

print("\nProcessing complete! Balancing dataset now...")

# Combine into large arrays
X_raw = np.vstack(all_x)
y_raw = np.array(all_y)

# --- 3. BALANCE & ACCURACY ---
idx_n = np.where(y_raw == 0)[0]
idx_v = np.where(y_raw == 1)[0]

print(f"\nRaw Counts -> Normal (N): {len(idx_n)} | Abnormal (V): {len(idx_v)}")

# Downsampling to 50/50 split for maximum accuracy
np.random.seed(42)
idx_n_balanced = np.random.choice(idx_n, size=len(idx_v), replace=False)
balanced_indices = np.concatenate([idx_n_balanced, idx_v])
np.random.shuffle(balanced_indices)

X_final = X_raw[balanced_indices]
y_final = y_raw[balanced_indices]

# --- 4. SAVE CLEAN DATA ---
np.save('X_final.npy', X_final)
np.save('y_final.npy', y_final)

print(f"--- SUCCESS ---")
print(f"Final Balanced Dataset: {len(y_final)} total beats.")
print(f"Saved as X_final.npy and y_final.npy")