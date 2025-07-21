import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load test CSV file
test_df = pd.read_csv('C:\python\sign_language_proj\data\sign_mnist_test.csv')

# Separate features and labels (assuming labels in last column)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Reshape input to 28x28x1 for model
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Normalize pixel values (if your model was trained on normalized data)
X_test /= 255.0

# Encode labels if categorical
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# One-hot encode labels
num_classes = len(np.unique(y_test_encoded))
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Load your trained Keras model
model = tf.keras.models.load_model('C:\python\sign_language_proj\model\sign_language_model.h5')

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=2)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict and classification report
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
