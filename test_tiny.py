import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("--- TINY TEST STARTING ---", flush=True)

# 1. Generate 100 heartbeats (180 samples each)
X_tiny = np.random.random((100, 180, 1))
y_tiny = np.random.randint(0, 2, (100, 1))

# 2. Build your exact Hardware Accelerator Architecture
model = models.Sequential([
    layers.Conv1D(8, kernel_size=5, activation='relu', input_shape=(180, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train for just 2 epochs
print("Starting Tiny Training...", flush=True)
model.fit(X_tiny, y_tiny, epochs=2, batch_size=4, verbose=1)

print("\n--- SUCCESS! YOUR ENVIRONMENT IS WORKING ---")