import pandas as pd
import numpy as np

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