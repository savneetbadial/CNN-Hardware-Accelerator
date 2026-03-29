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

print("--- Starting Training Pipeline ---", flush=True)

# --- 2. DATA LOADING ---
PILOT_MODE = False
try:
    X = np.load('X_final.npy')
    y = np.load('y_final.npy')

    if PILOT_MODE:
        X, y = X[:1000], y[:1000]
        print("!!! PILOT MODE ACTIVE !!!")

    X = X.reshape(X.shape[0], 180, 1)
    print(f"Loaded {len(X)} samples.", flush=True)
except FileNotFoundError:
    print("ERROR: Data files not found.")
    sys.exit()

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. SET ARCHITECTURE ---
model = models.Sequential([
    layers.Input(shape=(180, 1)),
    layers.Conv1D(8, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAINING CONFIGURATION ---
EPOCHS = 70

print(f"\n--- Starting Training for {EPOCHS} Epochs ---", flush=True)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1  # Changed to 1 to see the progress bar again
)

# --- 6. EVALUATION & VISUALIZATION ---
print("\n--- Generating Performance Metrics ---", flush=True)

y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype("int32")

# Metrics calculation
acc, prec, rec, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test,
                                                                                                   y_pred), f1_score(
    y_test, y_pred)

print("\n" + "=" * 35)
print(f"   PERFORMANCE AT {EPOCHS} EPOCHS")
print("=" * 35)
print(f" Accuracy:  {acc:.4%}")
print(f" Precision: {prec:.4%}")
print(f" Recall:    {rec:.4%}")
print(f" F1-Score:  {f1:.4%}")
print("=" * 35)

# Save the resulting model
model.save('ecg_model_latest.h5')
print(f"\nSUCCESS: Model saved as 'ecg_model_latest.h5'")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'V-Beat'],
            yticklabels=['Normal', 'V-Beat'])
plt.title(f'Confusion Matrix')
plt.show()