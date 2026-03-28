import setuptools
import sys

sys.modules['distutils'] = setuptools._distutils

import os

# 1. ENVIROMENT FIXES (Prevents macOS hangs)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Missing library: {e}. Please run 'pip install tensorflow numpy scikit-learn matplotlib seaborn'")
    sys.exit()

print("--- Starting Training Pipeline ---", flush=True)

# --- 2. SELECT DATA PORTION ---
# Set PILOT_MODE to True for a fast 1,000-sample test
# Set PILOT_MODE to False for the full 14,258-sample training
PILOT_MODE = True

try:
    print("Loading X_final.npy and y_final.npy...", end=" ", flush=True)
    X = np.load('X_final.npy')
    y = np.load('y_final.npy')
    print("Done.", flush=True)

    if PILOT_MODE:
        sample_size = 1000
        X = X[:sample_size]
        y = y[:sample_size]
        print(f"!!! PILOT MODE ACTIVE: Using only {sample_size} samples !!!", flush=True)
    else:
        print(f"FULL MODE ACTIVE: Using all {len(X)} samples.", flush=True)

    # Reshape for Conv1D: (Samples, Time Steps, Features)
    X = X.reshape(X.shape[0], 180, 1)

except FileNotFoundError:
    print("\nERROR: Data files not found. Run script.py first!")
    sys.exit()

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. HARDWARE-READY CNN ARCHITECTURE ---
model = models.Sequential([
    # Layer 1: 8 Filters, Kernel 5 (Matches hardware budget)
    layers.Conv1D(8, kernel_size=5, activation='relu', input_shape=(180, 1)),
    layers.MaxPooling1D(pool_size=2),

    # Layer 2: Classifier
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')  # Binary: 0=Normal, 1=V-Beat
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAINING LOOP ---
print("\n--- Beginning Training ---", flush=True)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 6. EVALUATION & VISUALIZATION ---
print("\n--- Generating Performance Metrics ---", flush=True)
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'V-Beat'], yticklabels=['Normal', 'V-Beat'])
plt.title('Confusion Matrix: ECG Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Save the model weights
model.save('ecg_model.h5')
print("\nSUCCESS: Model saved as 'ecg_model.h5'", flush=True)

plt.show()  # Note: Script stays open until you close the plot window