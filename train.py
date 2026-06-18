import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

print("=" * 60)
print("MNIST GPU TRAINING DEMO")
print("=" * 60)

print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))

# Load Dataset
print("\nLoading dataset...")
df = pd.read_csv("train.csv")

print(f"Dataset Shape: {df.shape}")

# Split Features and Labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Normalize
X = X.astype("float32") / 255.0

# Reshape for CNN
X = X.reshape(-1, 28, 28, 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting Training...\n")

start = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

end = time.time()

print(f"\nTraining Time: {end-start:.2f} seconds")

loss, accuracy = model.evaluate(X_test, y_test)

print("\n" + "=" * 60)
print(f"Test Accuracy: {accuracy:.4f}")
print("=" * 60)

model.save("mnist_model.keras")

print("\nModel saved as mnist_model.keras")