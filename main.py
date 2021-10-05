from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys
import warnings
import argparse

from build_lstm import build_lstm
from process_data import process_pretrain_data
from process_data import process_data
from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

INPUT_LENGTH = 12

""" Input the path to data files (csv)
"""

# Get all available files named by sensor ids with .csv extension
data_path = '/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors'

all_sensor_files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '.csv' in f]
print(f'We have {len(all_sensor_files)} sensors available.')
