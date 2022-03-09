# code for pre-training

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import sys
from datetime import datetime
import math
import csv
import pickle
import pandas as pd
import numpy as np
from build_lstm import build_lstm
from build_gru import build_gru

from process_data import process_pretrain_data
from process_data import process_pretrain_data_multi

import argparse

from keras.models import load_model
import tensorflow as tf


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

parser.add_argument('-rp', '--resume_dir_path', type=str, default=None, help='provide the dir path containing the pretrained h5 files to continue pretraining')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM network')
parser.add_argument('-dp', '--data_path', type=str, default='/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors', help='dataset path')
parser.add_argument('-p', '--pretrain_percentage', type=float, default=1.00, help='MUST SPECIFY `pre_train_percentage` to be non-0 for the pretrain function to run.')
parser.add_argument('-b', '--batch', type=int, default=256, help='batch number for the pretrain dataset')
parser.add_argument('-e', '--epoch', type=int, default=150, help='epoch number for pretrain')
parser.add_argument('-m', '--model', type=str, default='lstm', help='Model to choose - lstm or gru')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in 2 layers')
parser.add_argument('-ff', '--num_feedforward', type=int, default=12, help='number of feedforward predictions, used to set up the number of the neurons in the last layer of the model and the number of chained predictions (usually it has to be equal to -il)')
parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/device_outputs_Preprocessed_V1.1", help='base folder path to store running logs and h5 files')


args = parser.parse_args()
args = args.__dict__

dataset_path = args['data_path']
hidden_neurons = args['hidden_neurons']
logs_base_folder = args['logs_base_folder']

resume_dir_path = args["resume_dir_path"]

if resume_dir_path:
    log_files_folder_path = str(Path(resume_dir_path).parent.absolute())
    with open(f"{log_files_folder_path}/vars_record.pkl", 'rb') as f:
        vars_record = pickle.load(f)
    model_chosen = vars_record['model']
    pretrain_config = vars_record['pretrain_config']
    pretrain_percentage= vars_record['pretrain_percentage']
    INPUT_LENGTH = vars_record['INPUT_LENGTH']
    feed_forward = vars_record['feed_forward']
    logs_base_folder = vars_record['logs_base_folder']
else:
    vars_record = {}
    model_chosen = args['model']
    vars_record['model'] = model_chosen
    
    pretrain_config = {"batch": args['batch'], "epochs":  args['epoch']}
    pretrain_percentage = args['pretrain_percentage']
    INPUT_LENGTH = args['input_length']
    feed_forward = args['num_feedforward']
    vars_record['pretrain_config'] = pretrain_config
    vars_record['pretrain_percentage'] = pretrain_percentage
    vars_record['INPUT_LENGTH'] = INPUT_LENGTH
    vars_record['feed_forward'] = feed_forward
    vars_record['logs_base_folder'] = logs_base_folder
    # create log folder indicating by current running date and time
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_files_folder_path = f"{logs_base_folder}/{date_time}_pretrain_{model_chosen}"
    os.makedirs(log_files_folder_path, exist_ok=True)
    with open(f"{log_files_folder_path}/vars_record.pkl", 'wb') as f:
        pickle.dump(vars_record, f)

if model_chosen == 'lstm':
    build_model = build_lstm
elif model_chosen == 'gru':
    build_model = build_gru
else:
    sys.exit(f"Model specification error - must be 'lstm' or 'gru', but got {args['model']}.")

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
      # data_index = int(math.floor( (num_lines-1) // INPUT_LENGTH * pretrain_percentage )*INPUT_LENGTH)
      if num_lines < INPUT_LENGTH:
          sys.exit(f'Sample count({num_lines-1}) cannot be less than the INPUT_LENGTH({INPUT_LENGTH}) of the NN model.')
      data_index = int((num_lines-1) * pretrain_percentage)
      pretrain_data = pd.read_csv(file_path, nrows = data_index, encoding='utf-8').fillna(0)
      post_pretrain_data_index[sensor_file] = data_index
      pretrain_datasets.append((sensor_file, pretrain_data))

  # return pd.concat(pretrain_datasets, axis=0), post_pretrain_data_index
  return pretrain_datasets, post_pretrain_data_index

# pretrain the model (part of the run_pretrain())
def pretrain_model(model_path, X_train, y_train, log_files_folder_path, epoch, seq, batch, is_multi):
    """train the model
    
    # Arguments
        model: Model, NN model to train. In this project we always use LSTM.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        log_files_folder_path: specify directory to store log files
        pretrain_config: Dict, parameter for train.
    """    
    model = load_model(model_path)
    hist = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=1,
        validation_split=0.05)
    
    # save model weights and loss to file
    model_file_path = f'{log_files_folder_path}/{is_multi}_pretrain_{epoch}_{seq}.h5'
    model.save(model_file_path)
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{log_files_folder_path}/loss.csv', encoding='utf-8', index=False)
    return model_file_path

def run_pretrain(log_files_folder_path, pretrain_config, pretrain_percentage, all_sensor_files, dataset_path, INPUT_LENGTH, resume_path):
  # build pretrain_dataset
  pretrain_datasets, post_pretrain_data_index = build_pretrain_dataset(pretrain_percentage, INPUT_LENGTH, all_sensor_files, dataset_path)
  post_pretrain_data_index_saved_path = f'{log_files_folder_path}/post_pretrain_data_index.pkl'
  with open(post_pretrain_data_index_saved_path, 'wb') as f:
      pickle.dump(post_pretrain_data_index, f)
  # process data
  processed_pretrain_datasets = []
  processed_pretrain_datasets_multi = []
  for pretrain_dataset in pretrain_datasets:
      (sensor_file, pretrain_data) = pretrain_dataset
      X_train, y_train = process_pretrain_data(pretrain_data, INPUT_LENGTH)
      X_train_multi, y_train_multi = process_pretrain_data_multi(pretrain_data, INPUT_LENGTH, feed_forward)
      X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
      X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], X_train_multi.shape[1], 1))
      processed_pretrain_datasets.append((sensor_file, X_train, y_train))
      processed_pretrain_datasets_multi.append((sensor_file, X_train_multi, y_train_multi))
  # build pretrain model
  model_to_pretrain = build_model([INPUT_LENGTH, hidden_neurons, hidden_neurons, 1])
  model_to_pretrain_multi = build_model([INPUT_LENGTH, hidden_neurons, hidden_neurons, feed_forward])
  model_to_pretrain.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
  model_to_pretrain_multi.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
  if not resume_dir_path:
      model_file_path = f'{log_files_folder_path}/single_pretrain_0.h5'
      model_file_path_multi = f'{log_files_folder_path}/multi_pretrain_0.h5' # single and multi refer to the number of output neurons, being either 1 or multiple
      model_to_pretrain.save(model_file_path)
      model_to_pretrain_multi.save(model_file_path_multi)
      starting_epoch = 1
      seq = 1
  else:
      # get the starting_epoch
      # sort h5 files reversely to get the latest epoch number. To make it easy, retrain the last epoch and go from there
      sorted_h5_files = sorted(os.listdir(resume_dir_path,key=lambda x: int(x.split('_')[-1]), reverse=True))
      starting_epoch = int(sorted_h5_files[0].split('_')[-1])
      # seq corresponds to sensor_file sequence. To make it easy, always start from 1
      seq = 1
      model_file_path = f'{log_files_folder_path}/single_pretrain_{starting_epoch}_{seq}.h5'
      model_file_path_multi = f'{log_files_folder_path}/multi_pretrain_{starting_epoch}_{seq}.h5'
      
    #   if seq == len(): 
    #       starting_epoch += 1
    #       seq = 1
    #   else:
    #       seq = int(seq) + 1
      
      
  # begin training
  for epoch in range(starting_epoch, pretrain_config["epochs"] + 1):
      iter_control = range(seq - 1, len(processed_pretrain_datasets))
      for dataset_iter in iter_control:
          
          processed_pretrain_dataset = processed_pretrain_datasets[dataset_iter]
          processed_pretrain_dataset_multi = processed_pretrain_datasets_multi[dataset_iter]
          (sensor_file, X_train, y_train) = processed_pretrain_dataset
          (sensor_file, X_train_multi, y_train_multi) = processed_pretrain_dataset_multi
          
          print(f"Training epoch {epoch} csv file {seq}/{len(processed_pretrain_datasets)} {sensor_file}...")
          print("Training model with single output neural.")
          model_file_path = pretrain_model(model_file_path, X_train, y_train, log_files_folder_path, epoch, seq, pretrain_config["batch"], "single")
          print(f"Training model with {feed_forward} output neural.")
          model_file_path_multi = pretrain_model(model_file_path_multi, X_train_multi, y_train_multi, log_files_folder_path, epoch, seq, pretrain_config["batch"], "multi")
          seq += 1
      seq = 1

  print(f"The path to the files of the pretrained models are located at {model_file_path} and {model_file_path_multi}")

# Import data files (csv)
all_sensor_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and '.csv' in f]
# build pretrain dataset and model
run_pretrain(log_files_folder_path, pretrain_config, pretrain_percentage, all_sensor_files, dataset_path, INPUT_LENGTH, args["resume_dir_path"])