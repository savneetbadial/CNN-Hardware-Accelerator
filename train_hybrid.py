import setuptools
import sys

# Maintain environment stability
sys.modules['distutils'] = setuptools._distutils
import os

# ENVIRONMENT SETTINGS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import pywt  # Discrete Wavelet Transforms

print("--- Starting Hybrid CNN + Transformer ---", flush=True)

# DATA LOADING & MORPHOLOGICAL DSP PROCESSING

try:
    X_raw = np.load('X_final.npy')
    y = np.load('y_final.npy')
    if len(X_raw.shape) == 3:
        X_raw = X_raw.squeeze(-1)
    print(f"Loaded {X_raw.shape[0]} samples from data repository.", flush=True)
except FileNotFoundError:
    print("ERROR: Finalized data arrays not found.")
    sys.exit()

print("Executing 4-Level Daubechies-8 (db8) Wavelet Decomposition...", flush=True)
dwt_features = []
for sample in X_raw:
    # Extract structural detail coefficients to pinpoint QRS anomalies
    coeffs = pywt.wavedec(sample, 'db8', level=4)
    flattened_coeffs = np.concatenate([coeffs[0], coeffs[1], coeffs[2]])
    dwt_features.append(flattened_coeffs)
X_dwt = np.array(dwt_features)

print("Isolating Blind Sources via FastICA...", flush=True)
ica = FastICA(n_components=18, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_raw)

# Combine into a single morphological array
X_morphology = np.hstack((X_dwt, X_ica))

print("Compressing Dimensionality via PCA (Targeting 26 Principal Components)...", flush=True)
pca = PCA(n_components=26, random_state=42)
X_pca_features = pca.fit_transform(X_morphology)

# Format raw input for the 1D-CNN temporal sequence branch
X_signal = X_raw.reshape(X_raw.shape[0], 180, 1)

# Parallel Train/Test Splitting
X_sig_train, X_sig_test, X_pca_train, X_pca_test, y_train, y_test = train_test_split(
    X_signal, X_pca_features, y, test_size=0.2, random_state=42
)



# DUAL-INPUT HYBRID CNN + TRANSFORMER ARCHITECTURE

def build_hybrid_network():
    #  BRANCH A: Temporal 1D-CNN + Self-Attention Pipeline
    sig_input = layers.Input(shape=(180, 1), name="Raw_ECG_Signal_Input")
    cnn = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(sig_input)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(cnn)

    # Project spatial maps into Transformer internal dimensions
    d_model = 64
    proj = layers.Dense(d_model)(cnn)

    # Tiny Transformer Multi-Head Self-Attention Layer
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=d_model)(proj, proj)
    attn_normalized = layers.LayerNormalization()(attn + proj)
    flat_signal = layers.GlobalAveragePooling1D()(attn_normalized)

    #  BRANCH B: Hand-Crafted PCA Morphology Vector
    pca_input = layers.Input(shape=(26,), name="Engineered_PCA_Features_Input")
    dense_pca = layers.Dense(32, activation='relu')(pca_input)

    #  MULTI-MODAL FEATURE FUSION
    fused = layers.concatenate([flat_signal, dense_pca])
    shared = layers.Dense(32, activation='relu')(fused)
    dropout = layers.Dropout(0.3)(shared)
    output = layers.Dense(1, activation='sigmoid', name="Output_Classifier")(dropout)

    return models.Model(inputs=[sig_input, pca_input], outputs=output)


model = build_hybrid_network()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# TRAINING PIPELINE

print("\nTraining...", flush=True)
model.fit(
    x={"Raw_ECG_Signal_Input": X_sig_train, "Engineered_PCA_Features_Input": X_pca_train},
    y=y_train,
    validation_data=({"Raw_ECG_Signal_Input": X_sig_test, "Engineered_PCA_Features_Input": X_pca_test}, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

# Save high-level functional model architecture
model.save('hybrid_ecg_model.h5')


# HARDWARE PARAMETER EXTRACTION

print("\n--- Initiating Automated Hardware Parameter Extraction ---")

SCALE_FACTOR = 4096  # 2^12 for Q4.12 fixed-point representation


def float_to_hex16(value, scale):
    int_val = int(round(value * scale))
    int_val = max(-32768, min(32767, int_val))  # Enforce signed 16-bit register boundaries
    if int_val < 0:
        int_val = (1 << 16) + int_val  # Generate clean 16-bit Two's Complement
    return f"{int_val:04X}"


output_dir = 'hardware_hybrid_roms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parse functional graph layers and unroll parameters for SystemVerilog compilation
for layer in model.layers:
    all_weights = layer.get_weights()
    if len(all_weights) > 0:
        print(f"Serializing Layer: '{layer.name}' | Contains {len(all_weights)} sub-tensors")

        # Open a unified hexadecimal memory file for the entire layer
        with open(os.path.join(output_dir, f"{layer.name}_parameters.hex"), 'w') as f_out:
            # Flatten every sub-tensor sequentially (Weights, Biases, Attention Kernels, etc.)
            for tensor_idx, tensor in enumerate(all_weights):
                print(f"  -> Processing Tensor index {tensor_idx} with shape {tensor.shape}...")
                for val in tensor.flatten():
                    f_out.write(f"{float_to_hex16(val, SCALE_FACTOR)}\n")

print(f"\nSuccess: All quantized hexadecimal parameters compiled cleanly in '/{output_dir}'.")
