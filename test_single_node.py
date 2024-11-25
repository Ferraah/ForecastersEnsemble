import numpy as np
from joblib import Parallel, delayed
import psutil
import csv
import time
import sys
from forecaster_numpy import NumpyForecaster
from util import *

# read number of threads from command line
num_threads = int(sys.argv[1])
# Example input
X = np.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  
# Expected output for the first time step in the horizon
y = np.array([[0.1, 0.7]])  
# Random neural network parameters
W = np.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0, 1., 0., 1.]])  
# Random neural network bias
b = np.array([0.1]) 

# create an instance of the the forecaster class
forecaster = NumpyForecaster()

start = time.time()
# Number of forecasters
num_forecasters = 10000
# Standard deviation for the noise to initialize weights and biases
noise_std = 0.1 

aggregated_forecasting = []

horizon = 5
W_trained = []
b_trained = []

def run_forecaster(i: int, W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray, noise_std: float, horizon: int):
    # Initialize the weights and biases with noise
    W_noise = np.random.normal(0, noise_std, W.shape)
    b_noise = np.random.normal(0, noise_std, b.shape)

    W_init = W + W_noise
    b_init = b + b_noise

    # Train the model using the training loop
    W_trained, b_trained = forecaster.training_loop(forecaster.compute_gradients, 40, W_init, b_init, X, y)
    # Generate predictions
    y_predicted = forecaster.forecast(horizon, X, W_trained, b_trained)
    return y_predicted, W_trained, b_trained

# Parallelizing the loop using joblib
aggregated_forecasting = Parallel(n_jobs=num_threads)(delayed(run_forecaster)(i, W, b, X, y, noise_std, horizon) for i in range(num_forecasters))

end = time.time()
print(f"{end - start:.2f}\n")

# extract the weights and biases from the aggregated results
for i in range(num_forecasters):
    aggregated_forecasting[i], W_temp, b_temp = aggregated_forecasting[i]
    W_trained = np.append(W_trained, W_temp)
    b_trained = np.append(b_trained, b_temp)

# print(aggregated_forecasting)
store_biases_csv_joblib(b_trained, "./data/biases.csv")
store_weights_single_node_joblib(W_trained, num_forecasters, W.shape[1], W.shape[0], "./data/weights.csv")
write_aggregated_predictions_to_csv_joblib(aggregated_forecasting,'./data/aggregated_forecasting_joblib.csv')