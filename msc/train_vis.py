import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# 1. Load the Dataset
# Replace with your actual filename if it changes
df = pd.read_csv('eggplant_spectral_data.csv')

# 2. Filter for Visible Wavelengths ONLY (<= 700nm)
visible_channels = [
    'Ch_1', 'Ch_2', 'Ch_3', 'Ch_4', 'Ch_5', 'Ch_6',   # 410nm - 535nm (AS72653)
    'Ch_7', 'Ch_8', 'Ch_9',                           # 560nm - 645nm (AS72652)
    'Ch_13', 'Ch_14'                                  # 610nm, 680nm  (AS72651)
]

# Extract Features (X) and Labels (y)
X = df[visible_channels].values
y = df['Label'].values

# Encode Labels ('Healthy' = 0, 'Infested' = 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Split the Data (70% Train, 15% Validation, 15% Test)
# Step A: Split off 70% for training, leaving 30% for Val/Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

# Step B: Split the remaining 30% in half to get 15% Val and 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Data Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 4. Standardize the Features
# Neural networks perform best when input values are scaled around 0
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Reshape Data for 1D CNN
# 1D CNNs expect input shape: (Number of Samples, Number of Steps/Channels, Number of Features per step)
# Here, steps = 11 channels, features = 1 (the sensor reading itself)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 6. Build the 1D CNN Architecture
model = Sequential([
    # First Convolutional Block
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    
    # Second Convolutional Block
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    
    # Flatten the 1D feature maps into a standard 1D array
    Flatten(),
    
    # Fully Connected (Dense) Layers
    Dense(64, activation='relu'),
    Dropout(0.3), # Helps prevent the model from memorizing the training data (overfitting)
    Dense(1, activation='sigmoid') # Sigmoid for binary classification (0 or 1)
])

# 7. Compile the Model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 8. Train the Model
print("\n--- Starting Training ---")
history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,                  # Number of passes through the data
    batch_size=16,              # Number of samples processed before updating weights
    validation_data=(X_val_reshaped, y_val),
    verbose=1
)

# 9. Evaluate the Model on the Unseen Test Set
print("\n--- Testing Model on Unseen Data ---")
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

print(f"\nFinal Test Accuracy (Visible Only): {test_accuracy * 100:.2f}%")

# Save the model to a file
model.save('eggplant_vis_1dcnn.keras')
print("Model saved successfully as 'eggplant_vis_1dcnn.keras'")

import joblib
joblib.dump(scaler, 'spectral_scaler.pkl')
