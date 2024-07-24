import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
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
image_scaling = 90  # Example scaling value
batch_size = 16
epochs = 40
l2_lambda = 0.005
learning_rate = 0.0005

model_path = 'trained_model'

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'training_data')

# Verify paths
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory: {data_dir}")

if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"The directory {data_dir} does not exist.")

# Collect all subdirectories containing images
directories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Collect all image paths and labels
image_paths = [(directory, fname) for directory in directories for fname in os.listdir(directory) if fname.endswith('.png')]

# Callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# Initialize the CNNModel
cnn_model = CNNModel(input_shape=(image_scaling, image_scaling, 1), num_classes=2, learning_rate=learning_rate, l2_lambda=l2_lambda)

# Use the KFold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
mae_scores_dm = []
mae_scores_j0 = []
best_mae_dm = float('inf')
best_mae_j0 = float('inf')
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
        directories=[path[0] for path in train_paths],
        batch_size=batch_size,
        target_size=(image_scaling, image_scaling),
        subset='training',
        validation_split=0.2
    )

    val_generator = CustomDataGenerator(
        directories=[path[0] for path in val_paths],
        batch_size=batch_size,
        target_size=(image_scaling, image_scaling),
        subset='validation',
        validation_split=0.2
    )

    # Compile the model for regression
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
    val_predictions_dm = val_predictions[0].flatten()
    val_predictions_j0 = val_predictions[1].flatten()

    # Collect all true values from val_generator
    val_true_values_dm = []
    val_true_values_j0 = []
    for i in range(len(val_generator)):
        _, batch_y = val_generator[i]
        val_true_values_dm.extend(batch_y[0])
        val_true_values_j0.extend(batch_y[1])

    val_true_values_dm = np.array(val_true_values_dm)
    val_true_values_j0 = np.array(val_true_values_j0)

    if len(val_true_values_dm) == len(val_predictions_dm) and len(val_true_values_j0) == len(val_predictions_j0):
        val_mae_dm = mean_absolute_error(val_true_values_dm, val_predictions_dm)
        val_mae_j0 = mean_absolute_error(val_true_values_j0, val_predictions_j0)
        print(f"Validation MAE for fold {fold_no} (dm): {val_mae_dm}")
        print(f"Validation MAE for fold {fold_no} (j0): {val_mae_j0}")
        mae_scores_dm.append(val_mae_dm)
        mae_scores_j0.append(val_mae_j0)
        
        # Check if this model has the best validation MAE for both dm and j0
        if val_mae_dm < best_mae_dm and val_mae_j0 < best_mae_j0:
            best_mae_dm = val_mae_dm
            best_mae_j0 = val_mae_j0
            best_model = model
            best_history = history.history
    else:
        print(f"Mismatch in length for fold {fold_no}: {len(val_true_values_dm)} true values (dm), {len(val_predictions_dm)} predictions (dm)")
        print(f"Mismatch in length for fold {fold_no}: {len(val_true_values_j0)} true values (j0), {len(val_predictions_j0)} predictions (j0)")

    fold_no += 1

print(f"Mean MAE over all folds (dm): {np.mean(mae_scores_dm)}")
print(f"Mean MAE over all folds (j0): {np.mean(mae_scores_j0)}")

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
    plt.title(f'Training and Validation Loss for Best Fold (MAE: dm={best_mae_dm}, j0={best_mae_j0})')
    plt.legend()
    plt.savefig('best_fold_training_validation_loss.png')
    plt.show()

print(f"Validation MAE for best fold (dm): {best_mae_dm}")
print(f"Validation MAE for best fold (j0): {best_mae_j0}")
