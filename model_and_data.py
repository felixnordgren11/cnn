import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class CNNModel:
    def __init__(self, input_shape, num_classes=1, learning_rate=0.0005, l2_lambda = 0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.model_reg = self._build_model('regression')
        self.model_class = self._build_model('classification')

    def _build_model(self, model_type):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, kernel_regularizer=l2(self.l2_lambda)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(self.l2_lambda)),
            tf.keras.layers.Dropout(0.07),
            tf.keras.layers.Dense(self.num_classes)  # Output layer for the parameter(s)
        ])

        if model_type == 'classification':
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        elif model_type == 'regression':
            model.add(tf.keras.layers.Dense(1))  # Single output for regression

        return model

    def compile_model(self, model_type):
        if model_type == 'classification':
            self.model_class.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                     loss='sparse_categorical_crossentropy',
                                     metrics=['accuracy'])
            return self.model_class
        elif model_type == 'regression':
            self.model_reg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                   loss=tf.keras.losses.Huber(),
                                   metrics=['mae'])
            return self.model_reg

    def get_model(self, model_type):
        if model_type == 'classification':
            return self.model_class
        elif model_type == 'regression':
            return self.model_reg
        
        
# Function to preprocess (sharpen) the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1],
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

        
        
        
# Custom data generator class
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=16, target_size=(128, 128), subset='training', validation_split=0.2):
        self.directory = directory
        self.batch_size = batch_size
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

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = [self.load_image(self.filenames[i]) for i in batch_indexes]
        batch_y = [extract_parameter_from_filename(self.filenames[i]) for i in batch_indexes]
        #batch_x, batch_y = self.flatten_augmentations(batch_x, batch_y)
        return np.array(batch_x), np.array(batch_y)
    
    def reset(self):
        self.indexes = np.arange(len(self.filenames))
        np.random.shuffle(self.indexes)

    def load_image(self, filename):
        img_path = os.path.join(self.directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale mode
        img = cv2.resize(img, self.target_size)
        img = sharpen_image(img)  # Apply sharpening filter
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        
        #if self.subset in ['training', 'validation']:
        #    return self.augment_image(img)
        return img
    
    def augment_image(self, image):
        images = [image]
        for angle in [90, 180, 270]:
            rotated_image = self.rotate_image(image, angle)
            images.append(rotated_image)
        return images

    def rotate_image(self, image, angle):
        if angle == 90:
            return np.rot90(image, k=1)
        elif angle == 180:
            return np.rot90(image, k=2)
        elif angle == 270:
            return np.rot90(image, k=3)
        return image
    
    def flatten_augmentations(self, batch_x, batch_y):
        flattened_x, flattened_y = [], []
        for x, y in zip(batch_x, batch_y):
            if isinstance(x, list):  # If augmentations are present
                flattened_x.extend(x)
                flattened_y.extend([y] * len(x))
            else:
                flattened_x.append(x)
                flattened_y.append(y)
        return flattened_x, flattened_y
    
    
