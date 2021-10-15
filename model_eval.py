from process_data import process_data
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import csv
import math
import numpy as np
from keras.models import load_model
from tabulate import tabulate

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

parser.add_argument('-mp', '--model_path', type=str, default=None, help='the path of the model to evaluate')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM network')
parser.add_argument('-tp', '--test_percent', type=float, default=1.00, help='how much percent of the data to test')
parser.add_argument('-dp', '--data_path', type=str, default='/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors', help='dataset path')

args = parser.parse_args()
args = args.__dict__

INPUT_LENGTH = args['input_length']

dataset_path = args['data_path']

the_model = load_model(args['model_path'])
all_sensor_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and '.csv' in f]

error_values = {}
for sensor_file in all_sensor_files:
    sensor_id = sensor_file.split('.')[0]
    ''' processing data '''
    # data file path
    file_path = os.path.join(dataset_path, sensor_file)
    # count lines
    file = open(file_path)
    reader = csv.reader(file)
    num_lines = len(list(reader))
    # read data
    data_index = int((num_lines - 1) * (1 - args['test_percent']))
    test_data = pd.read_csv(file_path, skiprows = data_index, encoding='utf-8').fillna(0)
    _, _, X_test, y_test, scaler = process_data(np.empty(0), test_data, False, INPUT_LENGTH)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = the_model.predict(X_test)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(1, -1)[0]
    error_values[sensor_id] = {}
    error_values[sensor_id]['MAE'] = get_MAE(y_test, predictions)
    error_values[sensor_id]['MSE'] = get_MSE(y_test, predictions)
    error_values[sensor_id]['RMSE'] = get_RMSE(y_test, predictions)
    error_values[sensor_id]['MAPE'] = get_MAPE(y_test, predictions)

error_values_df = pd.DataFrame.from_dict(error_values)
print(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))