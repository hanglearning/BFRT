# code for pre-training

import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import math
import csv
import pickle
import pandas as pd
import numpy as np
from build_lstm import build_lstm
from process_data import process_pretrain_data

''' SET UP TRAINING CONFIG 

Note - 

1 - Specify the  `pretrain_config`, which includes the batch number.

2 - MUST SPECIFY `pre_train_percentage` to be non-0 for the pretrain function to run.

'''
pretrain_config = {"batch": 256, "epochs": 150}
pretrain_percentage = 1

INPUT_LENGTH = 12

# create log folder indicating by current running date and time
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/device_outputs_Preprocessed_V1.1/{date_time}_pretrain"

dataset_path = '/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors'

''' SET UP TRAINING CONFIG (END) '''



def build_pretrain_dataset(pretrain_percentage, INPUT_LENGTH, all_sensor_files, dataset_path):
  
  # init post_pretrain_data_index - used to record the data index that's next to the selected pretrained dataset. For example, if Indexes 1~100 is used for pretraining, then 101 is recorded
  post_pretrain_data_index = {}
  for sensor_file in all_sensor_files:
    post_pretrain_data_index[sensor_file] = 0
    
  pretrain_datasets = []
  for sensor_file in all_sensor_files:
    file_path = os.path.join(dataset_path, sensor_file)
    # count lines
    file = open(file_path)
    reader = csv.reader(file)
    num_lines = len(list(reader))
    # read data - multiplies of INPUT_LENGTH within the pretrain_percentage of data
    data_index = int(math.floor( (num_lines-1) // INPUT_LENGTH * pretrain_percentage )*INPUT_LENGTH)
    pretrain_data = pd.read_csv(file_path, nrows = data_index, encoding='utf-8').fillna(0)
    post_pretrain_data_index[sensor_file] = data_index
    pretrain_datasets.append(pretrain_data)

  return pd.concat(pretrain_datasets, axis=0), post_pretrain_data_index

# pretrain the model (part of the train_model())
def pretrain_model(model, X_train, y_train, log_files_folder_path, pretrain_config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train. In this project we always use LSTM.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        log_files_folder_path: specify directory to store log files
        pretrain_config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=pretrain_config["batch"],
        epochs=pretrain_config["epochs"],
        validation_split=0.05)
    
    
    # save model weights and loss to file
    model_file_path = f'{log_files_folder_path}/pretrain.h5'
    model.save(model_file_path)
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{log_files_folder_path}/loss.csv', encoding='utf-8', index=False)
    return model_file_path

def run_pretrain(log_files_folder_path, pretrain_config, pretrain_percentage, all_sensor_files, dataset_path, INPUT_LENGTH):
  with open(f'{log_files_folder_path}/pretrain_config.txt', 'w') as config_file:
    config_file.write(f"pretrain_config: {repr(pretrain_config)} + \n")
    config_file.write(f"INPUT_LENGTH: {repr(INPUT_LENGTH)} + \n")
    config_file.write(f"pretrain_percentage: {repr(pretrain_percentage)} + \n")
  # build pretrain_dataset
  pretrain_dataset, post_pretrain_data_index = build_pretrain_dataset(pretrain_percentage, INPUT_LENGTH, all_sensor_files, dataset_path)
  # process data
  X_train, y_train = process_pretrain_data(pretrain_dataset, INPUT_LENGTH)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  # build pretrain model
  model_to_pretrain = build_lstm([INPUT_LENGTH, 64, 64, 1])
  # begin training
  pretrained_model_file_path = pretrain_model(model_to_pretrain, X_train, y_train, log_files_folder_path, pretrain_config)

  post_pretrain_data_index_saved_path = f'{log_files_folder_path}/post_pretrain_data_index.pkl'
  with open(post_pretrain_data_index_saved_path, 'wb') as f:
      pickle.dump(post_pretrain_data_index, f)

  print(f"The path to the file of this pretrained model is located at {pretrained_model_file_path}")


# Import data files (csv)
all_sensor_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and '.csv' in f]
# build pretrain dataset and model
run_pretrain(log_files_folder_path, pretrain_config, pretrain_percentage, all_sensor_files, dataset_path, INPUT_LENGTH)