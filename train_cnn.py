import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load the data we saved from pre-processing
X = np.load('X_final.npy')
y = np.load('y_final.npy')

# Reshape for CNN (Samples, Length, Channels) -> (N, 180, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 2. Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Build a CNN
model = models.Sequential([
    # Layer 1: Conv1D - Only 8 filters to keep hardware small
    layers.Conv1D(8, kernel_size=5, activation='relu', input_shape=(180, 1)),
    layers.MaxPooling1D(pool_size=2),

    # Layer 2: Flatten and Output
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')  # 0 = Normal, 1 = Ventricular
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train!
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the model
model.save('ecg_model.h5')
print("Model trained and saved as ecg_model.h5")