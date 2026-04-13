import setuptools
import sys

# fix for Python + TensorFlow legacy dependencies
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

print("--- Starting Stabilized Deep Training Pipeline ---", flush=True)

# --- 2. DATA LOADING ---
PILOT_MODE = False
try:
    X = np.load('X_final.npy')
    y = np.load('y_final.npy')

    if PILOT_MODE:
        X, y = X[:1000], y[:1000]
        print("!!! PILOT MODE ACTIVE: Testing on 1000 samples !!!", flush=True)

    # Reshape for Conv1D: (Samples, Time Steps, Features)
    X = X.reshape(X.shape[0], 180, 1)
    print(f"Loaded {len(X)} samples successfully.", flush=True)
except FileNotFoundError:
    print("ERROR: 'X_final.npy' or 'y_final.npy' not found. Run script.py first!")
    sys.exit()

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. STABILIZED 6-LAYER ARCHITECTURE ---
# This design targets high accuracy while preventing the 'Gradient Explosion' seen previously.
model = models.Sequential([
    layers.Input(shape=(180, 1)),

    # Layer 1: Edge Detection
    layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=4),

    # Layer 2: Morphology Extraction
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),

    # Layer 3: High-Level Pattern Analysis
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),  # Hardware-friendly parameter reduction

    layers.Flatten(),

    # Layer 4: Regularization (Stops the model from overfitting)
    layers.Dropout(0.3),

    # Layer 5: Dense
    layers.Dense(64, activation='relu'),

    # Layer 6: Final Classification
    layers.Dense(1, activation='sigmoid')
])

# Use a lower Learning Rate (0.0001) to ensure smooth weight updates
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAINING CONFIGURATION ---
EPOCHS = 30  # Optimized to prevent the divergence seen at epoch 70

print(f"\n--- Starting Training for {EPOCHS} Epochs ---", flush=True)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 6. EVALUATION & VISUALIZATION ---
print("\n--- Generating Performance Metrics ---", flush=True)

y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype("int32")

# Final Metrics
acc, prec, rec, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), \
    recall_score(y_test, y_pred), f1_score(y_test, y_pred)

print("\n" + "=" * 35)
print(f"   FINAL PERFORMANCE METRICS")
print("=" * 35)
print(f" Accuracy:  {acc:.4%}")
print(f" Precision: {prec:.4%}")
print(f" Recall:    {rec:.4%}")
print(f" F1-Score:  {f1:.4%}")
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