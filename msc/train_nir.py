import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Load the Dataset
df = pd.read_csv('eggplant_spectral_data.csv')

# 2. Filter for NIR Wavelengths ONLY (> 700nm)
# AS72652 (705nm, 900nm, 940nm) and AS72651 (730nm, 760nm, 810nm, 860nm)
nir_channels = [
    'Ch_10', 'Ch_11', 'Ch_12', 
    'Ch_15', 'Ch_16', 'Ch_17', 'Ch_18'
]

# Extract Features (X) and Labels (y)
X = df[nir_channels].values
y = df['Label'].values

# Encode Labels ('Healthy' = 0, 'Infested' = 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Split the Data (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Data Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 4. Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Reshape Data for 1D CNN
# Now we have 7 channels instead of 11
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 6. Build the 1D CNN Architecture
model = Sequential([
    # First Convolutional Block
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    
    # Second Convolutional Layer (No pooling after this since we only have 7 starting steps)
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3), 
    Dense(1, activation='sigmoid')
])

# 7. Compile the Model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 8. Train the Model with Early Stopping
print("\n--- Starting Training (NIR Only) ---")
# Stop training if validation loss doesn't improve for 15 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_reshaped, y_train,
    epochs=150,                 # Set high; EarlyStopping will intervene
    batch_size=16,
    validation_data=(X_val_reshaped, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 9. Evaluate the Model on the Unseen Test Set
print("\n--- Testing Model on Unseen Data ---")
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

print(f"\nFinal Test Accuracy (NIR Only): {test_accuracy * 100:.2f}%")

# 10. Save the trained model
model.save('eggplant_nir_1dcnn.keras')
print("Model saved successfully as 'eggplant_nir_1dcnn.keras'")
