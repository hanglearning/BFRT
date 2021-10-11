import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import argparse

from keras.models import load_model

from build_lstm import build_lstm
from process_data import process_data
from model_training import train_baseline_model
from model_training import train_local_model
from chain_predict import chain_predict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

parser.add_argument('-rp', '--resume_path', type=str, default=None, help='provide the leftover log folder path to continue FL')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM network')
parser.add_argument('-dp', '--data_path', type=str, default='/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/traffic_data/Preprocessed_V1.1_4sensors', help='dataset path')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for FL')
parser.add_argument('-e', '--epoch', type=int, default=20, help='epoch number per comm round for FL')
parser.add_argument('-c', '--comm_rounds', type=int, default=240, help='number of comm rounds')


args = parser.parse_args()
args = args.__dict__

# determine if resume training
resume_training = False
if args['resume_path']:
	resume_training = True
	log_files_folder_path = args['resume_path']
else:
	''' Global Variables Set Up '''

	INPUT_LENGTH = args['input_length']

	# create log folder indicating by current running date and time
	date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
	log_files_folder_path = f"/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/device_outputs_Preprocessed_V1.1/{date_time}"
	
	dataset_path = args['data_path']

	# FL config 
	fl_config = {"batch": args['batch'], "epochs":  args['epoch']}
	communication_rounds = args['comm_rounds'] # 1 comm round = 1 hour

	vars_record = {} # to be stored and used for resuming training
	vars_record["INPUT_LENGTH"] = INPUT_LENGTH
	vars_record["dataset_path"] = dataset_path
	vars_record["fl_config"] = fl_config
	vars_record["communication_rounds"] = communication_rounds



''' Global Variables Set Up (END)'''

""" 1. Import data files (csv) """

# Get all available files named by sensor ids with .csv extension

all_sensor_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and '.csv' in f]
print(f'We have {len(all_sensor_files)} sensors available.')
vars_record["all_sensor_files"] = all_sensor_files

""" 2. Optional - Construct the pretrained model.

RUN THIS FUNCTION IN pretrain.py

"""

""" 3. Main functions for learning and predictions

(1) Data input of the learning process for both chained predictions and 1-step look_ahead prediction (as a baseline to compare with the errors from chained predictions).

Round 1 - Train on 1\~12 to learn the 13th point, so training set 1\~13

Round 2 - Train on 1\~24, learn 13th, 14th ... 24th

Round 3 - Train on 13\~36, learn 25th, 26th ... 36th

(2) Data input of the chained prediction process

Round 1 - Test on 13\~24

Round 2 - Test on 25\~36

Round 3 - Test on 37\~38

(3) Data input of the 1-step look_ahead prediction process

Round 1 - Testset input 1\~24 to predict on 13\~24

Round 2 - Testset input 13\~36 to predict on 25\~36

Round 3 - Testset input 25\~48 to predict on 37\~48

NOTE - Naive iterating over data. Will not deal with potential repeated or missing data.

"""

STARTING_ROUND = 1
if resume_training:
	with open(f"{log_files_folder_path}/vars_record.pkl", 'rb') as f:
		vars_record = pickle.load(f)
	dataset_path = vars_record["dataset_path"]
	fl_config = vars_record["fl_config"]
	communication_rounds = vars_record["communication_rounds"]
	all_sensor_files = vars_record["all_sensor_files"]
	starting_data_index = vars_record["starting_data_index"]
	INPUT_LENGTH = vars_record["INPUT_LENGTH"]
	# load starting round
	with open(f'{log_files_folder_path}/rounds_done.txt', 'r') as f:
		STARTING_ROUND = int(f.readlines()[-1]) + 1
	# load global model
	global_model = load_model(f'{log_files_folder_path}/globals/round_{STARTING_ROUND}.h5')
	# load all_sensor_predicts
	with open(f"{log_files_folder_path}/all_predicts.pkl", 'rb') as f:
		sensor_predicts = pickle.load(f)
	# load baseline models
	with open(f'{log_files_folder_path}/baseline_model_paths.pkl', 'rb') as f:
		baseline_models = pickle.load(f)
else:
	# set specific pretrained model, or set it to None to not use a pretrained model
	# pretrained_model_file_path = None
	pretrained_model_log_folder = '/content/drive/MyDrive/Traffic Prediction FedAvg Simulation/device_outputs_Preprocessed_V1.1/08262021_181808'
	pretrained_model_file_path = f'{pretrained_model_log_folder}/pretrain/pretrain.h5'
	if pretrained_model_file_path:
		with open(f"{pretrained_model_log_folder}/post_pretrain_data_index.pkl", 'rb') as f:
			post_pretrain_data_index = pickle.load(f)
	os.makedirs(f'{log_files_folder_path}/globals', exist_ok=True)

	# tf.compat.v1.disable_v2_behavior() # model trained in tf1
	# begin main function

	# train, FedAvg, Prediction (simulating real-time training)

	# init baseline models
	baseline_models = {}
	for sensor_file in all_sensor_files:
		sensor_id = sensor_file.split('.')[0]
		# create log directories for this sensor
		this_sensor_dir_path = f'{log_files_folder_path}/{sensor_id}'
		this_sensor_h5_baseline_model_path = f'{this_sensor_dir_path}/h5/baseline'
		this_sensor_h5_local_model_path = f'{this_sensor_dir_path}/h5/local'
		os.makedirs(this_sensor_h5_baseline_model_path, exist_ok=True)
		os.makedirs(this_sensor_h5_local_model_path, exist_ok=True)
		
		if pretrained_model_file_path:
			baseline_model = load_model(pretrained_model_file_path)
		else:
			baseline_model = build_lstm([INPUT_LENGTH, 64, 64, 1])
			model_file_path = f'{this_sensor_h5_baseline_model_path}/{sensor_id}_baseline_0.h5'
			baseline_model.save(model_file_path)
			baseline_models[sensor_file] = {}
			baseline_models[sensor_file]['model_file_path'] = model_file_path
			baseline_models[sensor_file]['this_sensor_dir_path'] = this_sensor_dir_path

	# init global model
	if pretrained_model_file_path:
		print("Starting FL with the pretrained model...")
		global_model = load_model(pretrained_model_file_path)
	else:
		global_model = build_lstm([INPUT_LENGTH, 64, 64, 1])
		global_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
		global_model_file_path = f'{log_files_folder_path}/globals/h5'
		os.makedirs(global_model_file_path, exist_ok=True)
		global_model.save(f'{global_model_file_path}/comm_0.h5')

	# init prediction records
	sensor_predicts = {}
	for sensor_file in all_sensor_files:
		sensor_predicts[sensor_file] = {}
		sensor_predicts[sensor_file]['baseline_chained'] = []
		sensor_predicts[sensor_file]['local_chained'] = []
		sensor_predicts[sensor_file]['global_chained'] = []
		sensor_predicts[sensor_file]['baseline_onestep'] = []
		sensor_predicts[sensor_file]['local_onestep'] = []
		sensor_predicts[sensor_file]['global_onestep'] = []
		sensor_predicts[sensor_file]['true'] = []

	# use the first sensor data created_time as the standard to ease simulation data slicing
	file_path = os.path.join(dataset_path, all_sensor_files[0])
	created_time_column = pd.read_csv(file_path, encoding='utf-8').fillna(0)['created_time']

	if pretrained_model_file_path:
		starting_data_index = post_pretrain_data_index[all_sensor_files[0]]
	else:
		starting_data_index = 0
	vars_record["starting_data_index"] = starting_data_index
	# store training config
	with open(f"{log_files_folder_path}/vars_record.pkl", 'wb') as f:
		pickle.dump(vars_record, f)

# print("starting_data_index", starting_data_index)

# init FedAvg vars
sample_size_each_communication_round = INPUT_LENGTH 

for round in range(STARTING_ROUND, communication_rounds + 1):
	local_model_weights = []
	global_weights = global_model.get_weights()
	print(f"Simulating comm round {round}...")
	if round == 1:
		training_data_starting_index = starting_data_index
	else:
		training_data_starting_index = starting_data_index + (round - 2) * sample_size_each_communication_round
	starting_created_time = created_time_column.iloc[training_data_starting_index]
	X_train_records = {}
	X_test_records = {}
	for sensor_file in all_sensor_files:
		sensor_id = sensor_file.split('.')[0]
		
		''' processing data '''
		# data file path
		file_path = os.path.join(dataset_path, sensor_file)
		# read data
		whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
		# get training data slicing indexes
		# try:
			# training_data_starting_index = whole_data[whole_data['created_time'] == starting_created_time].index[0]
			# change it to naively iterate csv dates
		# except:
		#   print(f"Data error - Sensor {sensor_id} does not have the row with created_time {starting_created_time}.")
		#   # sys.exit(0)
		#   continue
		if round == 1:
			training_data_ending_index = training_data_starting_index + sample_size_each_communication_round
		else:
			training_data_ending_index = training_data_starting_index + sample_size_each_communication_round * 2 - 1

		debug_text = f"DEBUG INFO: {sensor_id} now uses its own data starting at {training_data_starting_index} to {training_data_ending_index}\n"
		print(debug_text)
		
		# get test data slicing indexes
		# only use the labels of the test data to compare with chained (feed-forward) predictions
		if round == 1:
			chained_test_data_starting_index = training_data_ending_index
			onestep_test_data_starting_index = training_data_starting_index
		else:
			chained_test_data_starting_index = training_data_ending_index + 1
			onestep_test_data_starting_index = training_data_starting_index + sample_size_each_communication_round
		chained_test_data_ending_index = chained_test_data_starting_index + sample_size_each_communication_round - 1
		onestep_test_data_ending_index = onestep_test_data_starting_index + sample_size_each_communication_round * 2 - 1
		# slice training data
		train_data = whole_data[training_data_starting_index: training_data_ending_index + 1]
		# slice test data (the next sample_size_each_communication_round amount of data)
		chained_test_data = whole_data[chained_test_data_starting_index: chained_test_data_ending_index + 1]
		onestep_test_data = whole_data[onestep_test_data_starting_index: onestep_test_data_ending_index + 1]
		# process data
		# X_test won't be used in chained predictions
		X_train, y_train, _, y_test, scaler = process_data(train_data, chained_test_data, True, INPUT_LENGTH)
		X_train, y_train, X_test_onestep, y_test_onestep, scaler = process_data(train_data, onestep_test_data, False, INPUT_LENGTH)
		X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
		X_test_onestep = np.reshape(X_test_onestep, (X_test_onestep.shape[0], X_test_onestep.shape[1], 1))
		X_train_records[sensor_file] = X_train
		X_test_records[sensor_file] = X_test_onestep

		''' begin training '''
		print(f"{sensor_id} now training on row {training_data_starting_index} to {training_data_ending_index}...")
		# train baseline model
		print(f"{sensor_id} training baseline model..")
		new_baseline_model_path = train_baseline_model(round, baseline_models[sensor_file]['model_file_path'], X_train, y_train, sensor_id, baseline_models[sensor_file]['this_sensor_dir_path'], fl_config)
		baseline_models[sensor_file]['model_file_path'] = new_baseline_model_path
		# train local model
		print(f"{sensor_id} training local model..")
		new_local_model_path, new_local_model_weights = train_local_model(round, global_weights, X_train, y_train, sensor_id, baseline_models[sensor_file]['this_sensor_dir_path'], fl_config, INPUT_LENGTH)  
		# record local model
		local_model_weights.append(new_local_model_weights)

		''' predictions '''
		print(f"{sensor_id} now predicting on 3 models...")
		baseline_model = load_model(new_baseline_model_path)
		local_model = load_model(new_local_model_path)
		''' Onestep predictions '''
		# import pdb
		# pdb.set_trace()
		baseline_predicted = baseline_model.predict(X_test_onestep)
		baseline_predicted = scaler.inverse_transform(baseline_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		local_predicted = local_model.predict(X_test_onestep)
		local_predicted = scaler.inverse_transform(local_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['baseline_onestep'].append((round,baseline_predicted))
		sensor_predicts[sensor_file]['local_onestep'].append((round,local_predicted))
		''' chain predictions '''
		baseline_predicted = chain_predict(baseline_model, X_train, INPUT_LENGTH)
		baseline_predicted = scaler.inverse_transform(baseline_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		local_predicted = chain_predict(local_model, X_train, INPUT_LENGTH)
		local_predicted = scaler.inverse_transform(local_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['baseline_chained'].append((round,baseline_predicted))
		sensor_predicts[sensor_file]['local_chained'].append((round,local_predicted))
		''' true data '''
		y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['true'].append((round,y_test))
		
	
	''' Simulate FedAvg '''
	global_weights = np.mean(local_model_weights, axis=0)
	global_model.set_weights(global_weights)
	global_model.save(f'{log_files_folder_path}/globals/round_{round}.h5')
	# Predict by global model
	for sensor_file, sensor_attrs in sensor_predicts.items():
		print(f"Simulating {sensor_file} FedAvg...")
		# try:
			# global_predicted = global_model.predict(X_train_records[sensor_file])
		''' onestep prediction '''
		global_predicted = global_model.predict(X_test_records[sensor_file])
		global_predicted = scaler.inverse_transform(global_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		sensor_predicts[sensor_file]['global_onestep'].append((round,global_predicted))
		''' chain prediction '''
		# print(X_train_records[sensor_file])
		global_predicted = chain_predict(global_model, X_train_records[sensor_file], INPUT_LENGTH)
		# except:
		#   print(f'Data error - {sensor_file} does not contain data point {starting_created_time}.')
		#   # sys.exit(1)
		#   continue
		global_predicted = scaler.inverse_transform(global_predicted.reshape(-1, 1)).reshape(1, -1)[0]
		# print("global_predicted.shape", global_predicted.shape)
		sensor_predicts[sensor_file]['global_chained'].append((round,global_predicted))
	
	predictions_record_saved_path = f'{log_files_folder_path}/all_predicts.pkl'
	with open(predictions_record_saved_path, 'wb') as f:
		pickle.dump(sensor_predicts, f)

	''' Record Baseline Models Paths '''
	with open(f'{log_files_folder_path}/baseline_model_paths.pkl', 'wb') as f:
		pickle.dump(baseline_models, f)

	''' Record Resume Round'''
	with open(f'{log_files_folder_path}/rounds_done.txt', 'a+') as f:
		f.write(f'{round}\n')
 
	