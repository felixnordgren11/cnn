# main_script.py
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Subset
from model import CNNModel, CustomDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure the environment is properly configured for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_classes = 1  # We are predicting a single parameter value. Can be changed to predict multiple values later on. For example if we add the J value
image_scaling = 90  # Reduces the amount of weights needing to be trained, I think this size is a sweet spot to not lose too many features
batch_size = 16  # (16) gave an ok result, could try lowering to introduce more noise
epochs = 40
learning_rate = 0.0005  # Adjusted learning rate for Adam optimizer

# Paths
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(script_dir, 'training_data')



if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"The directory {data_dir} does not exist.")

# Define the model parameters
input_shape = (image_scaling, image_scaling)  # Example shape, adjust as necessary

# Collect all image paths and labels
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]

# Initialize the CNNModel
cnn_model = CNNModel(input_shape=input_shape, num_classes=num_classes).to(device)

# Use the KFold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
mae_scores = []
best_mae = float('inf')
best_model_state_dict = None

for train_index, val_index in kf.split(image_paths):
    print(f"Training fold {fold_no}...")

    train_paths = [image_paths[i] for i in train_index]
    val_paths = [image_paths[i] for i in val_index]

    train_dataset = CustomDataset(
        directory=data_dir,
        target_size=(image_scaling, image_scaling),
        subset='training',
        validation_split=0.2
    )

    val_dataset = CustomDataset(
        directory=data_dir,
        target_size=(image_scaling, image_scaling),
        subset='validation',
        validation_split=0.2
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Compile the regression model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        cnn_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
            labels = labels.view(-1).to(device).float()

            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        cnn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_predictions = []
            val_true_values = []
            for images, labels in val_loader:
                images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
                labels = labels.view(-1).to(device).float()
                outputs = cnn_model(images)
                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_true_values.extend(labels.cpu().numpy())
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    val_mae = mean_absolute_error(val_true_values, val_predictions)
    print(f"Validation MAE for fold {fold_no}: {val_mae}")
    mae_scores.append(val_mae)

    # Check if this model has the best validation MAE
    if val_mae < best_mae:
        best_mae = val_mae
        best_model_state_dict = cnn_model.state_dict()

    fold_no += 1

print(f"Mean MAE over all folds: {np.mean(mae_scores)}")

# Save the best model
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'best_fold_model.pth')
    print("Best model saved.")
