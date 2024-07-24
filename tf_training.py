import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.preprocessing.image import img_to_array
from model_and_data import CNNModel, CustomDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure the environment is properly configured for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'{len(gpus)} GPU(s) found and configured.')
    except RuntimeError as e:
        print(e)

# Parameters
num_classes = 1  # We are predicting a single parameter value. Can be changed to predict multiple values later on.
image_scaling = 90 # Reduces the amount of weights needing to be trained
batch_size = 16 # (16) gave an ok result, could try lowering to introduce more noise
epochs = 40 # Maybe decrease even, the loss increased a bit after around 30 epochs so probably good to keep it below that.
l2_lambda = 0.005 # Higher gives a bad result, maybe try lowering as well? But prob. not necessary



model_path = 'trained_model'

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'training_data', 'layer_1_5')

# Verify paths
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory: {data_dir}")

if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"The directory {data_dir} does not exist.")

# Define the model parameters
input_shape = (image_scaling, image_scaling, 1)  # Example shape, adjust as necessary

# Collect all image paths and labels
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]

# Callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# Initialize the CNNModel
cnn_model = CNNModel(input_shape=input_shape, num_classes=num_classes, learning_rate=0.0005)

# Use the KFold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
mae_scores = []
best_mae = float('inf')
best_model = None
best_history = None

# Lists to store training and validation losses for each fold
all_train_losses = []
all_val_losses = []

for train_index, val_index in kf.split(image_paths):
    print(f"Training fold {fold_no}...")
    
    train_paths = [image_paths[i] for i in train_index]
    val_paths = [image_paths[i] for i in val_index]
    
    train_generator = CustomDataGenerator(
        directory=data_dir,
        batch_size=batch_size,
        target_size=(image_scaling, image_scaling),
        subset='training',
        validation_split=0.2
    )

    val_generator = CustomDataGenerator(
        directory=data_dir,
        batch_size=batch_size,
        target_size=(image_scaling, image_scaling),
        subset='validation',
        validation_split=0.2
    )

    # Compile the regression model
    model = cnn_model.compile_model('regression')

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping]
    )

    # Append training and validation losses for this fold
    all_train_losses.append(history.history['loss'])
    all_val_losses.append(history.history['val_loss'])
    
    # Validate the model
    val_predictions = model.predict(val_generator)
    val_predictions = val_predictions.flatten()

    # Collect all true values from val_generator
    val_true_values = []
    for i in range(len(val_generator)):
        _, batch_y = val_generator[i]
        val_true_values.extend(batch_y)

    val_true_values = np.array(val_true_values)

    if len(val_true_values) == len(val_predictions):
        val_mae = mean_absolute_error(val_true_values, val_predictions)
        print(f"Validation MAE for fold {fold_no}: {val_mae}")
        mae_scores.append(val_mae)
        
        # Check if this model has the best validation MAE
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_history = history.history
    else:
        print(f"Mismatch in length for fold {fold_no}: {len(val_true_values)} true values, {len(val_predictions)} predictions")

    fold_no += 1

print(f"Mean MAE over all folds: {np.mean(mae_scores)}")

# Save the best model
if best_model is not None:
    best_model.save('best_fold_model.h5')
    print("Best model saved.")

# Padding sequences to the same length
max_epochs = max(len(losses) for losses in all_train_losses)

padded_train_losses = np.array([np.pad(losses, (0, max_epochs - len(losses)), 'constant', constant_values=np.nan) for losses in all_train_losses])
padded_val_losses = np.array([np.pad(losses, (0, max_epochs - len(losses)), 'constant', constant_values=np.nan) for losses in all_val_losses])

# Calculate average losses per epoch across all folds
avg_train_losses = np.nanmean(padded_train_losses, axis=0)
avg_val_losses = np.nanmean(padded_val_losses, axis=0)

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Training Loss')
plt.plot(avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

# Plot the training and validation losses for the best fold
if best_history is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(best_history['loss'], label='Training Loss (Best Fold)')
    plt.plot(best_history['val_loss'], label='Validation Loss (Best Fold)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for Best Fold (MAE: {best_mae})')
    plt.legend()
    plt.savefig('best_fold_training_validation_loss.png')
    plt.show()

print(f"Validation MAE for best fold: {best_mae}")
