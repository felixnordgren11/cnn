import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model_and_data import CustomDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import r2_score
#from keras.utils import plot_model


# Load the saved model
model = tf.keras.models.load_model('best_fold_model.h5')

TF_ENABLE_ONEDNN_OPTS=0


# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(script_dir, 'test_data')
num_files = len(os.listdir(test_data_dir))

# Parameters
num_classes = 1  # We are predicting a single parameter value. Can be changed to predict multiple values later on. For example if we add the J value
image_scaling = 90 # Reduces the amount of weights needing to be trained, i think this size is a sweet spot to not lose too many features
batch_size = num_files # (16) gave an ok result, could try lowering to  introduce more noise


test_generator = CustomDataGenerator(
    directory=test_data_dir,
    batch_size=batch_size,
    target_size=(image_scaling, image_scaling),
    subset=None,
    validation_split=0.0  # No validation split for test data
)

test_generator.reset()
# Evaluate the model
test_loss, test_mae = model.evaluate(test_generator)
print(f'Test MAE: {test_mae}')

# Predicting on test data
test_generator.reset()
predictions = model.predict(test_generator)
true_values = []
for i in range(len(test_generator)):
    _, batch_y = test_generator[i]
    true_values.extend(batch_y)

predictions = predictions.flatten()
true_values = np.array(true_values)

r2 = r2_score(true_values, predictions)
print(f'R² Value: {r2}')

ae = [(predictions[i]-true_values[i]) for i in range(len(predictions))]

for i in range(len(ae)):
    print(np.abs(ae[i]))
    
    # Print true values and predictions
for true_value, prediction in zip(true_values, predictions):
    print(f"True Value: {true_value}, Prediction: {prediction}")


# Ensure true_values and predictions have the same size
if len(true_values) == len(predictions):
    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red')  # Diagonal line for reference
    plt.savefig('true_vs_predicted_plot.png')  # Save the plot as a file
    #plt.show()
else:
    print("Mismatch in number of true values and predictions.")
    
    
#plot_model(model, to_file='model_vis.png')
