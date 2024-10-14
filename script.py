import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time

# Part 1: Setup and Data Preparation
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Part 2: Model Building
# Initialize and fit the logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Part 3: Report Model Metrics
# Model accuracy
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.4f}')

# Model size in bytes (weights only)
model_size = sum(param.nbytes for param in [model.coef_])
print(f'Original Model Size: {model_size} bytes')

# Inference time
start_time = time.time()
model.predict(X_test)
inference_time = time.time() - start_time
print(f'Original Inference Time: {inference_time:.6f} seconds')

# Part 4: Quantize the Model
def quantize_model(model, scale_factor=128):
    # Get the model's weights
    weights = model.coef_
    # Quantize weights to 8-bit
    quantized_weights = np.round(weights * scale_factor).astype(np.int8)
    return quantized_weights

quantized_weights = quantize_model(model)

# Part 5: Inference using the Quantized Model
def quantized_inference(X, quantized_weights):
    # Calculate the dot product with quantized weights
    logits = X @ quantized_weights.T
    predictions = np.argmax(logits, axis=1)
    return predictions

# Measure inference time for the quantized model
start_time = time.time()
quantized_predictions = quantized_inference(X_test, quantized_weights)
quantized_inference_time = time.time() - start_time

# Part 6: Report Quantized Model Metrics
# Quantized model accuracy
quantized_accuracy = np.mean(quantized_predictions == y_test)
print(f'Quantized Model Accuracy: {quantized_accuracy:.4f}')

# Quantized model size (weights only)
quantized_model_size = quantized_weights.nbytes
print(f'Quantized Model Size: {quantized_model_size} bytes')

# Quantized inference time
print(f'Quantized Inference Time: {quantized_inference_time:.6f} seconds')

# Model Size Comparison
print(f'Model Size Comparison:')
print(f'Original Model Size: {model_size} bytes')
print(f'Quantized Model Size: {quantized_model_size} bytes')

