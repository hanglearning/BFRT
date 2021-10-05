""" Functions for obtaining a pre-train model. Run this function in the next cell"""

# code for pre-training

# construct the pretrain dataset frame by percentage
import os
import math
import csv
import pandas as pd

# init post_pretrain_data_index - used to record the data index that's next to the selected pretrained dataset. For example, if Indexes 1~100 is used for pretraining, then 101 is recorded
post_pretrain_data_index = {}
for sensor_file in all_sensor_files:
  post_pretrain_data_index[sensor_file] = 0

def build_pretrain_dataset(pre_train_percentage, INPUT_LENGTH):
  pretrain_datasets = []
  for sensor_file in all_sensor_files:
    file_path = os.path.join(data_path, sensor_file)
    # count lines
    file = open(file_path)
    reader = csv.reader(file)
    num_lines = len(list(reader))
    # read data - multiplies of INPUT_LENGTH within the pre_train_percentage of data
    data_index = int(math.floor( (num_lines-1) // INPUT_LENGTH * pre_train_percentage )*INPUT_LENGTH)
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
    model_file_path = f'{pretrain_log_dir_path}/pretrain.h5'
    model.save(model_file_path)
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{pretrain_log_dir_path}/loss.csv', encoding='utf-8', index=False)
    return model_file_path