# corresponds to - Testing code: We need to be able to run inference/testing on the final 20% of the data for all 7 detectors


import argparse
import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from tabulate import tabulate

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from process_data import process_test_one_step
from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="inferencing code")

parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors', help='dataset path')
parser.add_argument('-ip', '--inferencing_percent', type=float, default=0.2, help='last percentage of the data for testing/inferencing')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')
parser.add_argument('-pr', '--plot_range', type=int, default=288, help='plot the learning curves of this number of the last predictions. default to 24*(60/5)=288 assuming 5 min resolution')

args = parser.parse_args()
args = args.__dict__

''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
scaler = config_vars["scaler"]
all_sensor_files = config_vars["all_sensor_files"]

with open(f'{logs_dirpath}/baseline_model_paths.pkl', 'rb') as f:
    baseline_models = pickle.load(f)
INPUT_LENGTH = config_vars['input_length']
time_res = args["time_resolution"]
plot_range = args["plot_range"]
''' load vars '''

plot_dir_path = f'{logs_dirpath}/plots/offline_inferences'
os.makedirs(plot_dir_path, exist_ok=True)

''' load data '''
# read test data for each sensor
test_data_dict = {}
individual_max_data_sample = 0
for sensor_file_iter in range(len(all_sensor_files)):
	sensor_file = all_sensor_files[sensor_file_iter]
	# data file path
	file_path = os.path.join(args['dataset_path'], sensor_file)
	# read data
	# count lines
	file = open(file_path)
	reader = csv.reader(file)
	num_lines = len(list(reader))
	read_line_start = int((num_lines-1) * (1 - args["inferencing_percent"]))
	individual_max_data_sample = read_line_start if read_line_start > individual_max_data_sample else individual_max_data_sample
	test_data = pd.read_csv(file_path, skiprows=[i for i in range(1,read_line_start+1)], encoding='utf-8').fillna(0) # need to keep header for skiprows https://thispointer.com/pandas-skip-rows-while-reading-csv-file-to-a-dataframe-using-read_csv-in-python/
	print(f'Skipped {read_line_start} lines of data from {sensor_file} and read the last percentage: {args["inferencing_percent"]} of data for inferencing. ({sensor_file_iter+1}/{len(all_sensor_files)})')
	test_data_dict[sensor_file] = test_data
''' load data '''

''' load global models '''
# get the lastest comm round
last_round = config_vars["last_round"]
single_global_model = load_model(f'{logs_dirpath}/globals/single_h5/comm_{last_round}.h5')
# multi_global_model = load_model(f'{logs_dirpath}/globals/multi_h5/comm_{last_round}.h5')
''' load global models '''

inference_record = {}
# baseline models predict/inference
for sensor_file_iter in range(len(all_sensor_files)):
    sensor_file = all_sensor_files[sensor_file_iter]
    sensor_id = sensor_file.split('.')[0]
    inference_record[sensor_id] = {}
    ''' load the latest baseline models '''
    single_baseline_model = load_model(baseline_models[sensor_file]['single_baseline_model_path'])
    # multi_baseline_model = load_model(baseline_models[sensor_file]['multi_baseline_model_path'])
    ''' process and reshape data '''
    X_test_onestep, y_test = process_test_one_step(test_data_dict[sensor_file], scaler, INPUT_LENGTH)
    X_test_onestep = np.reshape(X_test_onestep, (X_test_onestep.shape[0], X_test_onestep.shape[1], 1))
    ''' inferencing '''
    # baseline model
    print(f"{sensor_id} is inferencing on its test data by its baseline centralized model. ({sensor_file_iter+1}/{len(all_sensor_files)})")
    onestep_baseline_predictions = single_baseline_model.predict(X_test_onestep)
    inference_record[sensor_id]["single_baseline_model_inferences"] = scaler.inverse_transform(onestep_baseline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
    # global model
    print(f"{sensor_id} is inferencing on its test data by the global model. ({sensor_file_iter+1}/{len(all_sensor_files)})")
    onestep_global_predictions = single_global_model.predict(X_test_onestep)
    inference_record[sensor_id]["single_global_model_inferences"] = scaler.inverse_transform(onestep_global_predictions.reshape(-1, 1)).reshape(1, -1)[0]
    # record true data
    inference_record[sensor_id]["true"] = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    
def plot_and_save(inference_record):
    """Plot
    Plot the true data and predicted data.
    """
    for sensor_id, prediction_method in inference_record.items():
        plot_data = {}

        for model, predicts in prediction_method.items():

            plot_data[model] = {}
            plot_data[model]['x'] = range(len(predicts))
            plot_data[model]['y'] = predicts
      
        fig = plt.figure()
        ax = fig.add_subplot(111)
      
        x = plot_data['true']['x']
        # customize x axis labels as comm rounds
        # my_xticks = []
        # for x_ele in x:
        #     if (x_ele - 1) % INPUT_LENGTH == 0:
        #         my_xticks.append(x_ele//INPUT_LENGTH + 1)
        #     else:
        #         my_xticks.append('')

        # plt.xticks(x, my_xticks)


        # if plot_range > 22, there is an empty comm at the begining for x axis
        print(f"Plotting last {plot_range} comm rounds for {sensor_id}.")
        plotting_range = -int(60/time_res*plot_range)
        ax.plot(x[-plotting_range:], plot_data['true']['y'][plotting_range:], label='True Data')

        # ax.plot(x[-plotting_range:], plot_data['global_chained']['y'][plotting_range:], label='global_chained', color='darkgreen')
        ax.plot(x[-plotting_range:], plot_data['single_global_model_inferences']['y'][plotting_range:], label='global_onestep', color='#5a773a')
        # ax.plot(x[-plotting_range:], plot_data['global_multi']['y'][plotting_range:], label='global_multi', color='limegreen')

        # ax.plot(x[-plotting_range:], plot_data['baseline_chained']['y'][plotting_range:], label='baseline_chained', color='darkred')
        ax.plot(x[-plotting_range:], plot_data['single_baseline_model_inferences']['y'][plotting_range:], label='baseline_onestep', color='#ffb839')
        # ax.plot(x[-plotting_range:], plot_data['baseline_multi']['y'][plotting_range:], label='baseline_multi', color='lightsalmon')

        plt.legend()
        plt.grid(True)
        plt.xlabel('Timeline')
        plt.ylabel('Volume')
        plt.title(sensor_id)
        fig = plt.gcf()
        # fig.set_size_inches(228.5, 10.5)
        fig.set_size_inches(10.5, 3.5)
        print()
        plt.savefig(f'{plot_dir_path}/{sensor_id}.png', bbox_inches='tight', dpi=100)
        plt.show()
        
def calculate_errors(inference_record):
    error_values = {}
    for sensor_id, prediction_method in inference_record.items():
        error_values[sensor_id] = {}

        for model, predicts in prediction_method.items():
            if model != 'true':
                error_values[sensor_id][model] = {}
                error_values[sensor_id][model]['MAE'] = get_MAE(predicts, prediction_method['true'])
                error_values[sensor_id][model]['MSE'] = get_MSE(predicts, prediction_method['true'])
                error_values[sensor_id][model]['RMSE'] = get_RMSE(predicts, prediction_method['true'])
                error_values[sensor_id][model]['MAPE'] = get_MAPE(predicts, prediction_method['true'])
    for sensor_id, model in error_values.items():
        print(f'\nfor {sensor_id}')
        error_values_df = pd.DataFrame.from_dict(model)
        print(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))

plot_and_save(inference_record)
calculate_errors(inference_record)

