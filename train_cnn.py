import setuptools
import sys

sys.modules['distutils'] = setuptools._distutils
import os

# 1. ENVIRONMENT FIXES
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter  # NEW: For filtering

print("--- Starting Stabilized Deep Training Pipeline with DSP ---", flush=True)


# --- NEW: DSP FILTERING FUNCTION ---
def ecg_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """
    Standard ECG Bandpass: 0.5Hz - 45Hz.
    Removes Baseline Wander and Powerline Interference.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


# --- 2. DATA LOADING & FILTERING ---
PILOT_MODE = False
try:
    X = np.load('X_final.npy')
    y = np.load('y_final.npy')

    if PILOT_MODE:
        X, y = X[:1000], y[:1000]
        print("!!! PILOT MODE ACTIVE !!!")

    print(f"Applying Bandpass Filter to {len(X)} samples...", flush=True)
    # Apply filter to each 180-sample segment
    X_filtered = np.array([ecg_bandpass_filter(sample) for sample in X])

    # Reshape for Conv1D: (Samples, Time Steps, Features)
    X = X_filtered.reshape(X_filtered.shape[0], 180, 1)
    print(f"Filtering complete.", flush=True)
except FileNotFoundError:
    print("ERROR: Data files not found.")
    sys.exit()

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. STABILIZED 6-LAYER ARCHITECTURE ---
model = models.Sequential([
    layers.Input(shape=(180, 1)),
    layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=4),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAINING ---
EPOCHS = 30
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- 6. EVALUATION ---
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype("int32")

# Metrics
acc, prec, rec, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), \
    recall_score(y_test, y_pred), f1_score(y_test, y_pred)

print("\n" + "=" * 35)
print(f" Accuracy:  {acc:.4%}")
print(f" Recall:    {rec:.4%}")
print("=" * 35)


# --- 7. PLOTTING RESULTS (Stability Check) ---
acc_plot = history.history['accuracy']
val_acc_plot = history.history['val_accuracy']
loss_plot = history.history['loss']
val_loss_plot = history.history['val_loss']
epochs_range = range(1, len(acc_plot) + 1)

plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_plot, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc_plot, label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_plot, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss_plot, label='Validation Loss', linewidth=2)
plt.title('Model Loss (Binary Crossentropy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('cnn_training_results.png')  # Automatically save for your slides
print("\nSuccess: Plots saved as 'cnn_training_results.png'")
plt.show()

# Save the resulting model for hardware weight extraction
model.save('ecg_model_stabilized.h5')
print(f"Success: Model weights saved as 'ecg_model_stabilized.h5'")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'V-Beat'],
            yticklabels=['Normal', 'V-Beat'])
plt.title('Final Confusion Matrix')
plt.show()


# Weight extraction
print("\n--- Starting Hardware Parameter Extraction ---")

# Define fixed-point scaling factor (Q4.12 format: 1 sign bit, 3 integer bits, 12 fractional bits)
SCALE_FACTOR = 4096


# Convert floats to 16-bit Two's Complement Hexadecimal strings
def float_to_hex16(value, scale):
    int_val = int(round(value * scale))
    int_val = max(-32768, min(32767, int_val))  # Clip values to safe 16-bit boundaries

    if int_val < 0:
        int_val = (1 << 16) + int_val  # Convert to two's complement for negative numbers

    return f"{int_val:04X}"

# dedicated directory to store the hardware ROM initialization files
output_dir = 'hardware_roms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each layer in your exact model sequence
for layer in model.layers:
# Extract weights if the layer contains parameters (skips MaxPool, Flatten, and Dropout)
    if len(layer.get_weights()) > 0:
        weights, biases = layer.get_weights()
        print(f"Extracting Layer: '{layer.name}' | Weights Shape: {weights.shape} | Biases Shape: {biases.shape}")

    # Save weights as hex values for SystemVerilog $readmemh
        weight_file_path = os.path.join(output_dir, f"{layer.name}_weights.hex")
        with open(weight_file_path, 'w') as f_weight:
            for w in weights.flatten():
                f_weight.write(f"{float_to_hex16(w, SCALE_FACTOR)}\n")

    # 2. Save biases as hex values for SystemVerilog $readmemh
        bias_file_path = os.path.join(output_dir, f"{layer.name}_biases.hex")
        with open(bias_file_path, 'w') as f_bias:
            for b in biases.flatten():
                f_bias.write(f"{float_to_hex16(b, SCALE_FACTOR)}\n")

print(f"Success: All parameters quantized and saved as hexadecimal files in the '/{output_dir}' directory.")
