import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
train_df = pd.read_csv(r'C:\python\sign_language_proj\data\sign_mnist_train.csv')
test_df = pd.read_csv(r'C:\python\sign_language_proj\data\sign_mnist_test.csv')

# Separate labels and features
y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input: (samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Encode labels to 0-based integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Save the LabelEncoder for later use in evaluation
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ LabelEncoder saved as 'label_encoder.pkl'")

# One-hot encode labels
num_classes = len(le.classes_)
y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

# Print data info
print("✅ Data loaded and processed successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape (one-hot):", y_train_categorical.shape)
print("Number of classes:", num_classes)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test_categorical, verbose=2)
print(f"✅ Test accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('sign_language_model.h5')
print("✅ Model saved successfully as 'sign_language_model.h5'")
