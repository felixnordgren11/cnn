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
    def __init__(self, input_shape, num_classes=1, learning_rate=0.0005, l2_lambda=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda))(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = tf.keras.layers.Dropout(0.07)(x)
        
        output_dm = tf.keras.layers.Dense(1, name='dm_output')(x)  # Output for dm
        output_j0 = tf.keras.layers.Dense(1, name='j0_output')(x)  # Output for j0

        model = tf.keras.Model(inputs=inputs, outputs=[output_dm, output_j0])
        return model

    def compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss={'dm_output': tf.keras.losses.Huber(), 'j0_output': tf.keras.losses.Huber()},
                           metrics={'dm_output': 'mae', 'j0_output': 'mae'})
        return self.model

        
        
# Function to preprocess (sharpen) the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Helper function to extract parameters from filename
def extract_parameters_from_filename(directory, filename):
    match = re.search(r'mompix\.dm_(\d{5})\.Z001\.png', filename)
    if match:
        dm_value = float(match.group(1)) / 10000.0
        match_j0 = re.search(r'training_data/(\d{3})', directory)
        if match_j0:
            j0_value = float(match_j0.group(1)) / 1000.0
            return dm_value, j0_value
    return None, None

        
        
        
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directories, batch_size=16, target_size=(128, 128), subset='training', validation_split=0.2):
        self.directories = directories
        self.batch_size = batch_size
        self.target_size = target_size
        self.subset = subset
        self.filenames = []
        self.validation_split = validation_split
        for directory in directories:
            self.filenames.extend([(directory, f) for f in os.listdir(directory) if f.endswith('.png')])
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
        batch_y_dm = []
        batch_y_j0 = []
        for i in batch_indexes:
            dm_value, j0_value = extract_parameters_from_filename(self.filenames[i][0], self.filenames[i][1])
            batch_y_dm.append(dm_value)
            batch_y_j0.append(j0_value)
        
        return np.array(batch_x), [np.array(batch_y_dm), np.array(batch_y_j0)]
    
    def reset(self):
        self.indexes = np.arange(len(self.filenames))
        np.random.shuffle(self.indexes)

    def load_image(self, file_info):
        directory, filename = file_info
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale mode
        img = cv2.resize(img, self.target_size)
        img = sharpen_image(img)  # Apply sharpening filter
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        return img