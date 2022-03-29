import os
from os import listdir
from os.path import isfile, join
import sys
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import argparse
from copy import deepcopy

from keras.models import load_model

from build_lstm import build_lstm
from build_gru import build_gru
from model_training import train_baseline_model
from model_training import train_local_model
from chain_predict import chain_predict

from process_data import get_scaler
from process_data import process_train_data_single
from process_data import process_train_data_multi
from process_data import process_test_one_step
from process_data import process_test_chained
from process_data import process_test_multi_and_get_y_true

# remove some warnings
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

# arguments for system vars
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/traffic_data/', help='dataset path')
parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/BFRT_logs", help='base folder path to store running logs and h5 files')
parser.add_argument('-pm', '--preserve_historical_models', type=int, default=0, help='whether to preserve models from old communication rounds. Consume storage. Input 1 to preserve')

# arguments for resume training
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='provide the leftover log folder path to continue FL')

# arguments for pretrained models
# Readers: These two args are NOT used in CCGrid
parser.add_argument('-sp', '--single_output_pretrained_path', type=str, default=None, help='The single-output pretrained model file path')
parser.add_argument('-mp', '--multi_output_pretrained_path', type=str, default=None, help='The multi-output pretrained model file path')

# arguments for learning
parser.add_argument('-m', '--model', type=str, default='lstm', help='Model to choose - lstm or gru')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs_single', type=int, default=5, help='epoch number for models with single output per comm round for FL')
parser.add_argument('-em', '--epochs_multi', type=int, default=0, help='epoch number for models with multiple output per comm round for FL')
parser.add_argument('-ff', '--num_feedforward', type=int, default=12, help='number of feedforward predictions, used to set up the number of the last layer of the model and the number of chained predictions (usually it has to be equal to -il)')
parser.add_argument('-tp', '--train_percent', type=float, default=1.0, help='percentage of the data for training')

# arguments for federated learning
parser.add_argument('-c', '--comm_rounds', type=int, default=None, help='number of comm rounds, default aims to run until data is exhausted')
parser.add_argument('-ml', '--max_data_length', type=int, default=24, help='maximum data length for training in each communication round, simulating the memory space a sensor has')

args = parser.parse_args()
args = args.__dict__

''' Parse command line arguments '''

# determine if resume training
if args['resume_path']:
	logs_dirpath = args['resume_path']
	# load config variables
	with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
	all_sensor_files = config_vars["all_sensor_files"]
else:
	''' Global Variables Set Up '''

	# create log folder indicating by current running date and time
	date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
	logs_dirpath = f"{args['logs_base_folder']}/{date_time}_{args['model']}_input_{args['input_length']}_max_length_{args['max_data_length']}_epoch_{args['epochs_single']}"
	os.makedirs(logs_dirpath, exist_ok=True)
	# FL config
	model_chosen = args['model']
	if model_chosen == 'lstm':
		build_model = build_lstm
	elif model_chosen == 'gru':
		build_model = build_gru
	else:
		sys.exit(f"Model specification error - must be 'lstm' or 'gru', but got {args['model']}.")
	
	# save command line arguments
	config_vars = args
	
	""" Import data files (csv) """

	# Get all available files named by sensor ids with .csv extension

	all_sensor_files = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and '.csv' in f]
	print(f'We have {len(all_sensor_files)} sensors available.')
	
	config_vars["all_sensor_files"] = all_sensor_files
	config_vars['logs_dirpath'] = logs_dirpath
	# save config_vars for resuming training
	with open(f"{logs_dirpath}/config_vars.pkl", 'wb') as f:
		pickle.dump(config_vars, f)

''' Global Variables Set Up (END)'''


""" Optional - Construct the pretrained model.

RUN THIS FUNCTION IN pretrain.py

"""

""" Main functions for learning and predictions

NOTE - Naive iterating over data. Will not deal with potential repeated or missing data.

"""

# load data
# read whole data for each sensor
whole_data_dict = {}
whole_data_list = [] # to calculate scaler
individual_min_data_sample = float('inf') # to determine max comm rounds
for sensor_file_iter in range(len(all_sensor_files)):
	sensor_file = all_sensor_files[sensor_file_iter]
	# data file path
	file_path = os.path.join(config_vars['dataset_path'], sensor_file)
	# read data
	# count lines
	file = open(file_path)
	reader = csv.reader(file)
	num_lines = len(list(reader))
	read_to_line = int((num_lines-1) * config_vars["train_percent"])
	individual_min_data_sample = read_to_line if read_to_line < individual_min_data_sample else individual_min_data_sample
	whole_data = pd.read_csv(file_path, nrows=read_to_line, encoding='utf-8').fillna(0)
	print(f'Loaded {read_to_line} lines of data from {sensor_file} (percentage: {config_vars["train_percent"]}). ({sensor_file_iter+1}/{len(all_sensor_files)})')
	whole_data_dict[sensor_file] = whole_data
	whole_data_list.append(whole_data)

if args['resume_path']:
	# resume training
	with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
	# init build_model function
	build_model = build_lstm if config_vars["model"] == 'lstm' else build_gru
	# load starting round
	last_round = config_vars["last_round"]
	# to make it easy, retrain the last comm round for all models
	STARTING_ROUND = last_round + 1
	# load global model
	single_global_model = load_model(f'{logs_dirpath}/globals/single_h5/comm_{last_round}.h5')
	multi_global_model = load_model(f'{logs_dirpath}/globals/multi_h5/comm_{last_round}.h5')
	# load all_sensor_predicts
	with open(f"{logs_dirpath}/realtime_predicts.pkl", 'rb') as f:
		sensor_predicts = pickle.load(f)
	# load baseline model paths
	with open(f'{logs_dirpath}/baseline_model_paths.pkl', 'rb') as f:
		baseline_model_paths = pickle.load(f)
	# load global model paths
	with open(f'{logs_dirpath}/global_model_paths.pkl', 'rb') as f:
		global_model_paths = pickle.load(f)
	# other exposed vars
	# whole_data_dict = config_vars["whole_data_dict"]
	scaler = config_vars["scaler"]
	all_sensor_files = config_vars["all_sensor_files"]
else:
	STARTING_ROUND = 1
	# load pretrained models if specified
	single_pretrained_model = args['single_output_pretrained_path']
	multi_pretrained_model = args['multi_output_pretrained_path']
	if single_pretrained_model and multi_pretrained_model:
		pretrained_model_log_folder = str(single_pretrained_model.parent.absolute()) # or get it from multi_pretrained_model.parent.absolute()
		with open(f"{pretrained_model_log_folder}/post_pretrain_data_index.pkl", 'rb') as f:
			post_pretrain_data_index = pickle.load(f)
		starting_data_index = post_pretrain_data_index[all_sensor_files[0]]
	else:
		print("\nPretrained models read error/not provided. Training from scratch...")
		starting_data_index = 0
	# init global model
	global_model_paths = {}
	os.makedirs(f'{logs_dirpath}/globals', exist_ok=True)
	# assign pretrained models if specified
	if single_pretrained_model and multi_pretrained_model:
		print("Starting FL with the pretrained models...")
		single_global_model = load_model(single_pretrained_model)
		multi_global_model = load_model(multi_pretrained_model)
	else:
		# single model
		single_global_model = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], 1])
		single_global_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
		single_global_model_path = f'{logs_dirpath}/globals/single_h5'
		os.makedirs(single_global_model_path, exist_ok=True)
		single_global_model.save(f'{single_global_model_path}/comm_0.h5')
		# multi model
		multi_global_model = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], config_vars['num_feedforward']])
		multi_global_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
		multi_global_model_path = f'{logs_dirpath}/globals/multi_h5'
		os.makedirs(multi_global_model_path, exist_ok=True)
		multi_global_model.save(f'{multi_global_model_path}/comm_0.h5')
		# record global models for easy access
		global_model_paths[0] = {}
		global_model_paths[0]["single"] = f'{single_global_model_path}/comm_0.h5'
		global_model_paths[0]["multi"] = f'{multi_global_model_path}/comm_0.h5'
	# init baseline models
	baseline_model_paths = {}
	for sensor_file in all_sensor_files:
		sensor_id = sensor_file.split('.')[0]
		
		# create log directories for this sensor
		this_sensor_dirpath = f'{logs_dirpath}/{sensor_id}'
		
		# create log folders for baseline models
		h5_single_baseline_dirpath = f'{this_sensor_dirpath}/baseline_single_h5'
		h5_multi_baseline_dirpath = f'{this_sensor_dirpath}/baseline_multi_h5'
		os.makedirs(h5_single_baseline_dirpath, exist_ok=True)
		os.makedirs(h5_multi_baseline_dirpath, exist_ok=True)
  
		# create log folders for fl local models
		h5_single_local_dirpath = f'{this_sensor_dirpath}/local_single_h5'
		h5_multi_local_dirpath = f'{this_sensor_dirpath}/local_multi_h5'
		os.makedirs(h5_single_local_dirpath, exist_ok=True)
		os.makedirs(h5_multi_local_dirpath, exist_ok=True)
		
		# init baseline models
		# assign pretrained models if specified
		if single_pretrained_model and multi_pretrained_model:
			single_baseline_model = load_model(single_pretrained_model)
			multi_baseline_model = load_model(multi_pretrained_model)
		else:
			# create new models
			single_baseline_model = deepcopy(single_global_model)
			multi_baseline_model = deepcopy(multi_global_model)
			single_baseline_model_path = f'{h5_single_baseline_dirpath}/comm_0.h5'
			single_baseline_model.save(single_baseline_model_path)
			multi_baseline_model_path = f'{h5_multi_baseline_dirpath}/comm_0.h5'
			multi_baseline_model.save(multi_baseline_model_path)
		# record baseline models for easy access
		baseline_model_paths[sensor_file] = {}
		baseline_model_paths[sensor_file]['single_baseline_model_path'] = {}
		baseline_model_paths[sensor_file]['multi_baseline_model_path'] = {}
		baseline_model_paths[sensor_file]['single_baseline_model_path'][0] = single_baseline_model_path
		baseline_model_paths[sensor_file]['multi_baseline_model_path'][0] = multi_baseline_model_path
		baseline_model_paths[sensor_file]['this_sensor_dirpath'] = this_sensor_dirpath
	
	# init prediction records
	sensor_predicts = {}
	for sensor_file in all_sensor_files:
		sensor_predicts[sensor_file] = {}
		# baseline - centralized
		sensor_predicts[sensor_file]['baseline_onestep'] = []
		sensor_predicts[sensor_file]['baseline_chained'] = []
		sensor_predicts[sensor_file]['baseline_multi'] = []
		# local
		# sensor_predicts[sensor_file]['local_single'] = []
		# sensor_predicts[sensor_file]['local_chained'] = []
		# sensor_predicts[sensor_file]['local_multi'] = []
		# global
		sensor_predicts[sensor_file]['global_onestep'] = []
		sensor_predicts[sensor_file]['global_chained'] = []
		sensor_predicts[sensor_file]['global_multi'] = []
		# true
		sensor_predicts[sensor_file]['true'] = []

	# use the first sensor data created_time as the standard to ease simulation data slicing
	# file_path = os.path.join(args['dataset_path'], all_sensor_files[0])
	# created_time_column = pd.read_csv(file_path, encoding='utf-8').fillna(0)['created_time']
		
	# config_vars["created_time_column"] = created_time_column
	config_vars["starting_data_index"] = starting_data_index

	# get scaler
	# TODO - test if whole, and, get scaler from pretraining??
	scaler = get_scaler(pd.concat(whole_data_list))
	config_vars["scaler"] = scaler
	# store training config
	with open(f"{logs_dirpath}/config_vars.pkl", 'wb') as f:
		pickle.dump(config_vars, f)

# print("starting_data_index", starting_data_index)

# init FedAvg vars
INPUT_LENGTH = config_vars['input_length']
new_sample_size_per_comm_round = INPUT_LENGTH

# determine maximum comm rounds by the minimum number of data sample a device owned and the input_length
max_comm_rounds = individual_min_data_sample // INPUT_LENGTH - 2
# comm_rounds overwritten while resuming
if args['comm_rounds']:
	config_vars['comm_rounds'] = args['comm_rounds']
if config_vars['comm_rounds']:
	if max_comm_rounds > config_vars['comm_rounds']:
		print(f"\nNote: the provided dataset allows running for maximum {max_comm_rounds} comm rounds but the simulation is configured to run for {config_vars['comm_rounds']} comm rounds.")
		run_comm_rounds = config_vars['comm_rounds']
	elif config_vars['comm_rounds'] > max_comm_rounds:
		print(f"\nNote: the provided dataset ONLY allows running for maximum {max_comm_rounds} comm rounds, which is less than the configured {config_vars['comm_rounds']} comm rounds.")
		run_comm_rounds = max_comm_rounds
	else:
		run_comm_rounds = max_comm_rounds
else:
	print(f"\nNote: the provided dataset allows running for maximum {max_comm_rounds} comm rounds.")
	run_comm_rounds = max_comm_rounds
	
print(f"Starting FedAvg with total comm rounds {run_comm_rounds}...")
# tf.compat.v1.disable_v2_behavior() # model trained in tf1    
''' Local Training, FedAvg, Prediction (simulating real-time FedAvg) '''
# one comm round -> feed in one hour of new data (5 min resolution) if input_length = 12

# save used arguments as a text file for easy review
with open(f'{logs_dirpath}/args_used.txt', 'w') as f:
	f.write("Command line arguments used -\n")
	f.write(' '.join(sys.argv[1:]))
	f.write("\n\nAll arguments used -\n")
	for arg_name, arg in args.items():
		f.write(f'\n--{arg_name} {arg}')

for round in range(STARTING_ROUND, run_comm_rounds + 1):
	single_local_model_weights = []
	multi_local_model_weights = []
	single_global_weights = single_global_model.get_weights()
	multi_global_weights = multi_global_model.get_weights()
	print(f"Simulating comm round {round}/{run_comm_rounds} ({round/run_comm_rounds:.0%})...")
	# starting_created_time = created_time_column.iloc[training_data_starting_index] # now we assume data is preprocessed
	X_train_records = {}
	X_test_records = {}
	for sensor_file_iter in range(len(all_sensor_files)):
		sensor_file = all_sensor_files[sensor_file_iter]
		sensor_id = sensor_file.split('.')[0]
		''' Process traning data '''
		if round == 1:
			training_data_starting_index = starting_data_index
			training_data_ending_index = training_data_starting_index + new_sample_size_per_comm_round * 2 - 1
			# if it's round 1 and input_shape 12, need at least 24 training data points because the model at least needs 13 points to train.
			# Therefore,
			# round 1 -> 1~24 training points, predict with test 13~35 test points, 
   			# 1- 24， 2 - 36， 3 - 48， 4 - 60
		else:
			training_data_ending_index = (round + 1) * new_sample_size_per_comm_round - 1
			training_data_starting_index = training_data_ending_index - config_vars['max_data_length']
			if training_data_starting_index < 1:
				training_data_starting_index = 0
		whole_data = whole_data_dict[sensor_file]
		# slice training data
		train_data = whole_data[training_data_starting_index: training_data_ending_index + 1]

		# process training data
		X_train_single, y_train_single = process_train_data_single(train_data, scaler, INPUT_LENGTH)
		X_train_multi, y_train_multi = process_train_data_multi(train_data, scaler, INPUT_LENGTH, config_vars['num_feedforward'])
		debug_text = f"DEBUG INFO: {sensor_id} ({sensor_file_iter+1}/{len(all_sensor_files)}) now uses its own data starting at {training_data_starting_index} to {training_data_ending_index}\n"
		print(debug_text)
		
		''' Process test data '''
		
		test_data_starting_index = training_data_ending_index - new_sample_size_per_comm_round + 1
		test_data_ending_index_one_step = test_data_starting_index + new_sample_size_per_comm_round * 2 - 1
		test_data_ending_index_chained_multi = test_data_starting_index + new_sample_size_per_comm_round - 1
		
		# slice test data
		# test_data_onestep_multi is used for the one_step model
		# test_data_chained is used for the chained and the multi-output models
		test_data_onestep_multi = whole_data[test_data_starting_index: test_data_ending_index_one_step + 1]
		test_data_chained = whole_data[test_data_starting_index: test_data_ending_index_chained_multi + 1]

		# process data
		X_test_onestep, _ = process_test_one_step(test_data_onestep_multi, scaler, INPUT_LENGTH)
		X_test_chained = process_test_chained(test_data_chained, scaler, INPUT_LENGTH)
		X_test_multi, y_true = process_test_multi_and_get_y_true(test_data_onestep_multi, scaler, INPUT_LENGTH, config_vars['num_feedforward'])

		''' reshape data '''
		for train_data in ['X_train_single', 'X_train_multi', 'X_test_onestep', 'X_test_chained', 'X_test_multi']:
			vars()[train_data] = np.reshape(vars()[train_data], (vars()[train_data].shape[0], vars()[train_data].shape[1], 1))
	
		X_train_records[sensor_file] = {}
		X_train_records[sensor_file]['X_train_single'] = X_train_single
		X_train_records[sensor_file]['X_train_multi'] = X_train_multi
		X_test_records[sensor_file] = {}
		X_test_records[sensor_file]['X_test_onestep'] = X_test_onestep
		X_test_records[sensor_file]['X_test_chained'] = X_test_chained
		X_test_records[sensor_file]['X_test_multi'] = X_test_multi
		
		''' begin training '''
		print(f"{sensor_id} now training on row {training_data_starting_index} to {training_data_ending_index}...")
		
		''' train baseline model '''
		# single model
		print(f"{sensor_id} training single baseline model.. (1/4)")
		the_model = load_model(baseline_model_paths[sensor_file]['single_baseline_model_path'][round-1])
		new_single_baseline_model_path = train_baseline_model(the_model, round, X_train_single, y_train_single, sensor_id, baseline_model_paths[sensor_file]['this_sensor_dirpath'], "single", config_vars['batch'], config_vars['epochs_single'], config_vars['preserve_historical_models'])
		# multi model
		print(f"{sensor_id} training multi baseline model.. (2/4)")
		the_model = load_model(baseline_model_paths[sensor_file]['multi_baseline_model_path'][round-1])
		new_multi_baseline_model_path = train_baseline_model(the_model, round, X_train_multi, y_train_multi, sensor_id, baseline_model_paths[sensor_file]['this_sensor_dirpath'], "multi", config_vars['batch'], config_vars['epochs_multi'], config_vars['preserve_historical_models'])
		# record new baseline model paths
		baseline_model_paths[sensor_file]['single_baseline_model_path'][round] = new_single_baseline_model_path
		baseline_model_paths[sensor_file]['multi_baseline_model_path'][round] = new_multi_baseline_model_path
		
		''' train local model '''
		# single model
		print(f"{sensor_id} training single local model.. (3/4)")
		local_model = load_model(global_model_paths[round-1]["single"])
		new_single_local_model_path, new_single_local_model_weights = train_local_model(local_model, round, X_train_single, y_train_single, sensor_id, baseline_model_paths[sensor_file]['this_sensor_dirpath'], "single", config_vars['batch'], config_vars['epochs_single'], config_vars['preserve_historical_models'])
		# multi model
		print(f"{sensor_id} training multi local model.. (4/4)")
		local_model = load_model(global_model_paths[round-1]["multi"])
		new_multi_local_model_path, new_multi_local_model_weights = train_local_model(local_model, round, X_train_multi, y_train_multi, sensor_id, baseline_model_paths[sensor_file]['this_sensor_dirpath'], "multi", config_vars['batch'], config_vars['epochs_multi'], config_vars['preserve_historical_models'])
		# record local model
		single_local_model_weights.append(new_single_local_model_weights)
		multi_local_model_weights.append(new_multi_local_model_weights)
		''' baseline model predictions ''' # removed local models
		# load models
		single_baseline_model = load_model(new_single_baseline_model_path)
		multi_baseline_model = load_model(new_multi_baseline_model_path)
		''' (Single-output) Onestep Predictions '''
		print(f"{sensor_id} now predicting by its single baseline model (1/3)...")
		onestep_baseline_predictions = single_baseline_model.predict(X_test_onestep)
		onestep_baseline_predictions = scaler.inverse_transform(onestep_baseline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['baseline_onestep'].append((round,onestep_baseline_predictions))
		# local_predicted = local_model.predict(X_test_single)
		# local_predicted = scaler.inverse_transform(local_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		# sensor_predicts[sensor_file]['local_single'].append((round,local_predicted))
		''' (Single-output) Chained Predictions '''
		print(f"{sensor_id} now predicting by its chained baseline model (2/3)...")
		chained_baseline_predictions = chain_predict(single_baseline_model, X_test_chained, INPUT_LENGTH)
		chained_baseline_predictions = scaler.inverse_transform(chained_baseline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['baseline_chained'].append((round,chained_baseline_predictions))
		''' Multi-output Predictions '''
		print(f"{sensor_id} now predicting by its chained baseline model (3/3)...")
		multi_baseline_predictions = multi_baseline_model.predict(X_test_multi)
		multi_baseline_predictions = scaler.inverse_transform(multi_baseline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['baseline_multi'].append((round,multi_baseline_predictions))
		''' true data '''
		y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['true'].append((round,y_true))
		
	
	''' Simulate FedAvg '''
	# create single-output global model
	single_global_weights = np.mean(single_local_model_weights, axis=0)
	single_global_model.set_weights(single_global_weights)
	model_root_path = f'{logs_dirpath}/globals/single_h5'
	single_global_model.save(f'{model_root_path}/comm_{round}.h5')
	# clean model files if not preserve
	if not args["preserve_historical_models"]:
		filelist = [f for f in os.listdir(model_root_path) if not f.endswith(f'comm_{round}.h5') and not f.endswith(f'comm_{round-1}.h5')]
		for f in filelist:
			os.remove(os.path.join(model_root_path, f))
	# create multi-output global model
	multi_global_weights = np.mean(multi_local_model_weights, axis=0)
	multi_global_model.set_weights(multi_global_weights)
	multi_global_model.save(f'{logs_dirpath}/globals/multi_h5/comm_{round}.h5')
 
	model_root_path = f'{logs_dirpath}/globals/multi_h5'
	multi_global_model.save(f'{model_root_path}/comm_{round}.h5')
	# clean model files if not preserve
	if not args["preserve_historical_models"]:
		filelist = [f for f in os.listdir(model_root_path) if not f.endswith(f'comm_{round}.h5') and not f.endswith(f'comm_{round-1}.h5')]
		for f in filelist:
			os.remove(os.path.join(model_root_path, f))
 
	# store global model paths
	global_model_paths[round] = {}
	global_model_paths[round]["single"] = f'{logs_dirpath}/globals/single_h5/comm_{round}.h5'
	global_model_paths[round]["multi"] = f'{logs_dirpath}/globals/multi_h5/comm_{round}.h5'
	# Predict by global models
	sensor_count = 1
	for sensor_file, sensor_attrs in sensor_predicts.items():
		sensor_id = sensor_file.split('.')[0]
		print(f"Simulating {sensor_id} ({sensor_count}/{len(all_sensor_files)}) FedAvg...")
		# try:
			# global_predicted = global_model.predict(X_train_records[sensor_file])
		''' (Single-output) Onestep Predictions '''
		print(f"{sensor_id} now predicting by the single-output global model using onestep method.. (1/3)")
		single_global_predicted = single_global_model.predict(X_test_records[sensor_file]['X_test_onestep'])
		single_global_predicted = scaler.inverse_transform(single_global_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['global_onestep'].append((round,single_global_predicted))
		''' (Single-output) Chained Predictions '''
		print(f"{sensor_id} now predicting by the single-output global model using chained method.. (2/3)")
		# print(X_train_records[sensor_file])
		chained_global_predicted = chain_predict(single_global_model, X_test_records[sensor_file]['X_test_chained'], INPUT_LENGTH)
		# except:
		#   print(f'Data error - {sensor_file} does not contain data point {starting_created_time}.')
		#   # sys.exit(1)
		#   continue
		chained_global_predicted = scaler.inverse_transform(chained_global_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		# print("global_predicted.shape", global_predicted.shape)
		sensor_predicts[sensor_file]['global_chained'].append((round,chained_global_predicted))
		''' Multi-output Predictions '''
		print(f"{sensor_id} now predicting by the multi-output global model.. (3/3)")
		multi_global_predicted = multi_global_model.predict(X_test_records[sensor_file]['X_test_multi'])
		multi_global_predicted = scaler.inverse_transform(multi_global_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['global_multi'].append((round,multi_global_predicted))
		sensor_count += 1
	
	predictions_record_saved_path = f'{logs_dirpath}/realtime_predicts.pkl'
	with open(predictions_record_saved_path, 'wb') as f:
		pickle.dump(sensor_predicts, f)

	''' Record Baseline Model Paths '''
	with open(f'{logs_dirpath}/baseline_model_paths.pkl', 'wb') as f:
		pickle.dump(baseline_model_paths, f)
	
	''' Record Global Model Paths '''
	with open(f"{logs_dirpath}/global_model_paths.pkl", 'wb') as f:
		pickle.dump(global_model_paths, f)

	''' Record Resume Round'''
	config_vars["last_round"] = round
	with open(f"{logs_dirpath}/config_vars.pkl", 'wb') as f:
		pickle.dump(config_vars, f)
 
print(f"Simulation conclueded after comm rounds {round}.")