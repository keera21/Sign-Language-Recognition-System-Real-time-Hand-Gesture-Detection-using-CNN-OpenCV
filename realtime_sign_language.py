import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'sign_language_model.h5'))

# Open webcam (use 0 or 1 depending on your system)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Flip and draw region of interest (ROI) box
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = 300, 100, 550, 350  # yellow box region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Extract the ROI and preprocess
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(input_data, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_letter = chr(predicted_class_index + ord('A'))

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Sign Language Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
