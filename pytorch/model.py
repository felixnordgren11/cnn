import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'training_data')


# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (input_shape[0] // 16) * (input_shape[1] // 16), 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Function to preprocess (sharpen) the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# Helper function to extract parameter from filename
def extract_parameter_from_filename(filename):
    match = re.search(r'mompix\.dm_(\d{5})\.Z001\.png', filename)
    if match:
        parameter_value = float(match.group(1)) / 10000.0
        return parameter_value
    return None


# Custom Dataset
class CustomDataset(data.Dataset):
    def __init__(self, directory=data_dir, target_size=(128, 128), subset='training', validation_split=0.2):
        self.directory = directory
        self.target_size = target_size
        self.subset = subset
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.validation_split = validation_split
        self.n = len(self.filenames)
        self.indexes = np.arange(self.n)
        self.split_index = int(self.n * (1 - self.validation_split))
        if self.subset == 'training':
            self.indexes = self.indexes[:self.split_index]
        elif self.subset == 'validation':
            self.indexes = self.indexes[self.split_index:]
        np.random.shuffle(self.indexes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img_name = self.filenames[self.indexes[index]]
        img_path = os.path.join(self.directory, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = sharpen_image(image)
        image = self.transform(image)

        parameter = extract_parameter_from_filename(img_name)

        return image, torch.tensor(parameter)



# Data Loaders
def get_dataloader(directory, batch_size=16, target_size=(128, 128), validation_split=0.2):
    train_dataset = CustomDataset(directory, target_size, subset='training', validation_split=validation_split)
    val_dataset = CustomDataset(directory, target_size, subset='validation', validation_split=validation_split)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


# Model Training
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.0005, model_type='regression'):
    criterion = nn.MSELoss() if model_type == 'regression' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 1, images.size(2), images.size(3))  # Reshape to (batch_size*augments, 1, H, W)
            labels = labels.view(-1)  # Reshape to (batch_size*augments,)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 1, images.size(2), images.size(3))
                labels = labels.view(-1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return model


