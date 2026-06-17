import os
import numpy as np
import scipy.signal as signal
import pywt  # To handle the Daubechies-8 Wavelet Transform
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(" Starting Training (V2) ")

# DATA INGESTION

print("Loading clinical samples from data repository...")

X_raw_signal = np.load('X_final.npy').astype(np.float32)
y = np.load('y_final.npy').astype(np.float32)

print(f"Loaded raw signals with shape: {X_raw_signal.shape}")

print("Executing Zero-Phase Bidirectional Butterworth Filter (0.5Hz - 45Hz)...")


def apply_bidirectional_filter(data, fs=360.0):
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 45.0 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    return filtered_data


# Clear out baseline wander noise
X_signal_clean = apply_bidirectional_filter(X_raw_signal)

print("Parallel DSP: Computing 4-Level DWT (db8) across windows...")
# Extract wavelet coefficients to capture multi-resolution QRS frequency spikes
dwt_features = []
for i in range(X_signal_clean.shape[0]):
    coeffs = pywt.wavedec(X_signal_clean[i, :], 'db8', level=4)
    flat_coeffs = np.concatenate(coeffs)
    dwt_features.append(flat_coeffs)
X_dwt = np.array(dwt_features)

print("Compressing Wavelet coefficients to 26 PCA Components...")
# Compress the high-dimensional DWT vector pool down to 26 structural core features
pca = PCA(n_components=26, random_state=42)
X_pca_compressed = pca.fit_transform(X_dwt)

print("Extracting 18-Basis FastICA source profiles...")
# Extract explicit physiological independent source pathways from raw data
ica = FastICA(n_components=18, random_state=42, max_iter=500)
X_ica_features = ica.fit_transform(X_signal_clean)

print("Merging PCA and ICA...")
# Concatenate parallel features into a single matrix
X_morph_features = np.concatenate([X_pca_compressed, X_ica_features], axis=-1).astype(np.float32)
print(f"-> Successfully generated hand-crafted feature matrix: {X_morph_features.shape}")

# Prepare raw signal inputs for the 1D-CNN
X_signal = np.expand_dims(X_signal_clean, axis=-1)

# Parallel Train/Test Splitting (80/20 split)
X_sig_train, X_sig_test, X_morph_train, X_morph_test, y_train, y_test = train_test_split(
    X_signal, X_morph_features, y, test_size=0.2, random_state=42
)




print("\nBuilding Functional Architecture...")

# Raw Temporal Signal Processing
raw_signal_input = layers.Input(shape=(180, 1), name="Raw_ECG_Signal_Input")
x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu', name="conv1d_v2_1")(raw_signal_input)
x = layers.MaxPooling1D(pool_size=2, name="max_pooling1d_v2_1")(x)
x = layers.SpatialDropout1D(0.2, name="spatial_dropout_1")(x)

x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name="conv1d_v2_2")(x)
x = layers.SpatialDropout1D(0.2, name="spatial_dropout_2")(x)

# Project to Transformer internal space
x = layers.Dense(64, activation='relu', name="dense_projection")(x)

# Multi-Head Self-Attention Block
attention_out = layers.MultiHeadAttention(num_heads=2, key_dim=64, name="multi_head_attention_v2")(x, x)
x = layers.Add(name="residual_connection")([x, attention_out])
x = layers.LayerNormalization(name="layer_normalization_v2")(x)
flat_signal = layers.GlobalAveragePooling1D(name="global_average_pooling_v2")(x)

# Shape updated automatically to accept the total fused feature dimension (26 + 18 = 44)
morph_input = layers.Input(shape=(X_morph_features.shape[1],), name="Engineered_Morphological_Input")
y_dense = layers.Dense(64, activation='relu', name="morph_dense_1")(morph_input)
y_dense = layers.BatchNormalization(name="morph_batch_norm_1")(y_dense)
y_dense = layers.Dense(64, activation='relu', name="morph_dense_2")(y_dense)
y_dense = layers.Dense(32, activation='relu', name="morph_dense_3")(y_dense)

# --- Fusion & classification ---
fused_features = layers.Concatenate(name="symmetrical_fusion")([flat_signal, y_dense])
z = layers.Dense(32, activation='relu', name="dense_post_fusion")(fused_features)
z = layers.Dropout(0.3, name="classifier_dropout")(z)
output_classifier = layers.Dense(1, activation='sigmoid', name="Output_Classifier")(z)

model = models.Model(inputs=[raw_signal_input, morph_input], outputs=output_classifier)


# OPTIMIZATION & TRAINING LOOP

print("\nCommencing Training Loop...")

lr_decay = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    [X_sig_train, X_morph_train], y_train,
    validation_data=([X_sig_test, X_morph_test], y_test),
    epochs=15,
    batch_size=32,
    callbacks=[lr_decay]
)


# HARDWARE PARAMETER EXTRACTION

print("\n--- Initiating Hardware Parameter Extraction ---")

SCALE_FACTOR = 4096


def float_to_hex16(value, scale):
    int_val = int(round(value * scale))
    int_val = max(-32768, min(32767, int_val))
    if int_val < 0:
        int_val = (1 << 16) + int_val
    return f"{int_val:04X}"


output_dir = 'hardware_hybrid_v2_roms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for layer in model.layers:
    all_weights = layer.get_weights()
    if len(all_weights) > 0:
        print(f"Serializing Layer: '{layer.name}' | Sub-Tensors found: {len(all_weights)}")
        with open(os.path.join(output_dir, f"{layer.name}_parameters.hex"), 'w') as f_out:
            for tensor_idx, tensor in enumerate(all_weights):
                for val in tensor.flatten():
                    f_out.write(f"{float_to_hex16(val, SCALE_FACTOR)}\n")

print(f"\nSuccess: All quantized parameters compiled cleanly in '/{output_dir}'.")