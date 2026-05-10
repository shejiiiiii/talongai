import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.metrics import classification_report

# 1. Load the same dataset
df = pd.read_csv('eggplant_spectral_data.csv')

# 2. Filter for Visible Wavelengths ONLY
visible_channels = [
    'Ch_1', 'Ch_2', 'Ch_3', 'Ch_4', 'Ch_5', 'Ch_6',
    'Ch_7', 'Ch_8', 'Ch_9',
    'Ch_13', 'Ch_14'
]

X = df[visible_channels].values
y = df['Label'].values

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Recreate the EXACT same Test Split (Random state 42 ensures this matches)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 4. Standardize and Reshape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on train
X_test_scaled = scaler.transform(X_test)       # Transform test

X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 5. Load your saved model
model = tf.keras.models.load_model('eggplant_vis_1dcnn.keras')

# 6. Make Predictions
print("Evaluating model...")
y_pred_probs = model.predict(X_test_reshaped)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# 7. Print the Detailed Precision/Recall Report
report = classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_,
    digits=4
)

print("\n--- Detailed Classification Report ---")
print(report)
