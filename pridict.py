import numpy as np
import tensorflow as tf
import cv2
import pickle

# Paths
model_path = 'model/sign_language_model.h5'
encoder_path = 'model/label_encoder.pkl'
image_path = 'sample.png'  # Change extension if needed (e.g., sample.jpg)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load the label encoder
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Load and preprocess the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
if img is None:
    raise FileNotFoundError(f"❌ Image '{image_path}' not found!")

# Resize to match model input
img = cv2.resize(img, (28, 28))
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for model

# Predict
pred_probs = model.predict(img)
pred_index = np.argmax(pred_probs)
pred_label = label_encoder.inverse_transform([pred_index])[0]
confidence = np.max(pred_probs)

# Output
print(f"✅ Predicted Sign: {pred_label}")
print(f"🔍 Confidence: {confidence:.2f}")
