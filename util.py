import pandas as pd
import numpy as np
import csv

def store_predictions_csv(data: list, input_size: int, horizon: int, filename: str):
    # Given the data result list i want to store it in a csv file
    
    dim_headers = [f"dim{i}," for i in range(input_size)]
    header = "step,"+ "".join(dim for dim in dim_headers)
    header = header[:-1] + "\n"
    with open(filename, 'w') as file:
        file.write(header)
        for gathered_predictions in data:
            for prediction in gathered_predictions:
                for i, value in enumerate(prediction):
                    to_write = f"{i},"+"".join(f"{v}," for v in value)
                    to_write = to_write[:-2] + "\n"
                    file.write(to_write)
    file.close()
def store_biases_csv(data: list, filename: str):
    
    header = "forecaster, bias\n"
    id = 0
    with open(filename, 'w') as file:
        file.write(header)
        for biases in data:
            for bias in biases:
                file.write(f"{id},{bias[0]}\n")
                id += 1
    file.close()

def store_weights_csv(data: list, input_size: int, weights_size: int, filename: str):
    
    dim_headers = [f"w{i}{j}," for i in range(input_size) for j in range(weights_size)]
    header = "forecaster,"+ "".join(dim for dim in dim_headers)
    header = header[:-1] + "\n"
    forecaster = 0
    with open(filename, 'w') as file:
        file.write(header)
        for weights in data:
            for weight in weights:
                to_write = f"{forecaster},"
                for related_variable_weight in weight:
                    for value in related_variable_weight:
                        to_write += f"{value},"
                to_write = to_write[:-1] + "\n"
                forecaster += 1
                file.write(to_write)
    file.close()

def store_biases_csv_joblib(data: list, filename: str):
    
    header = "forecaster, bias\n"
    id = 0
    with open(filename, 'w') as file:
        file.write(header)
        for bias in data:
            file.write(f"{id},{bias}\n")
            id += 1
    file.close()
    
def store_weights_csv_joblib(data: list, input_size: int, weights_size: int, filename: str):
    dim_headers = [f"w{i}{j}," for i in range(input_size) for j in range(weights_size)]
    header = "forecaster,"+ "".join(dim for dim in dim_headers)
    header = header[:-1] + "\n"
    forecaster = 0
    with open(filename, 'w') as file:
        file.write(header)
        to_write = f"{forecaster},"
        for value in data:
            to_write += f"{value},"
        to_write = to_write[:-1] + "\n"
        forecaster += 1
        file.write(to_write)
        file.close()

def store_weights_single_node_joblib(W: list, num_forecasters: int, input_size: int, output_size: int, filename: str):
    """
    Stores the aggregated weights of multiple forecasters into a CSV file.

    Args:
        W (list): A flat list of weights where the first W.shape[0]*W.shape[1] entries belong to the first forecaster,
                  the next W.shape[0]*W.shape[1] entries to the second forecaster, and so on.
        num_forecasters (int): The number of forecasters.
        input_size (int): The number of input dimensions (rows of W for each forecaster).
        output_size (int): The number of output dimensions (columns of W for each forecaster).
        filename (str): The name of the output CSV file.
    """
    # Convert W to a numpy array
    W = np.array(W)

    # Validate the total size of W
    total_weights = input_size * output_size * num_forecasters
    if W.size != total_weights:
        raise ValueError(
            f"Mismatch in the size of W: expected {total_weights} elements, but got {W.size} elements."
        )

    # Reshape W to group weights by forecaster
    reshaped_W = W.reshape(num_forecasters, input_size, output_size)

    # Generate column headers
    columns = ['forecaster'] + [f'w{row}{col}' for row in range(input_size) for col in range(output_size)]

    # Prepare data for CSV
    data = []
    for forecaster_id in range(num_forecasters):
        forecaster_weights = reshaped_W[forecaster_id].flatten()
        data.append([forecaster_id] + forecaster_weights.tolist())

    # Write to CSV using 'with open'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write the header
        writer.writerows(data)  # Write the data rows

    print(f"CSV written to {filename}")

def write_aggregated_predictions_to_csv_joblib(aggregated_predictions, filename):
    # Open the CSV file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['step', 'dim0', 'dim1'])
        
        # Iterate over the forecasters
        for forecaster_id, forecaster in enumerate(aggregated_predictions):
            # For each forecaster, iterate over the 5 steps and their corresponding dimension values
            for step in range(forecaster.shape[0]):
                dim0, dim1 = forecaster[step]  # Get dim0 and dim1 for the current step
                writer.writerow([step, dim0, dim1])  # Write the data to the CSV file
