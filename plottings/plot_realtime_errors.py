# plot 4 errors during FL simulation


import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="inferencing code")

parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-et', '--error_type', type=str, default="MAE", help='error type to plot')
# parser.add_argument('-pr', '--plot_range', type=int, default=288, help='plot the learning curves of this number of the last predictions. default to 24*(60/5)=288 assuming 5 min resolution')

args = parser.parse_args()
args = args.__dict__

''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
all_sensor_files = config_vars["all_sensor_files"]

with open(f'{logs_dirpath}/realtime_predicts.pkl', 'rb') as f:
    realtime_predicts = pickle.load(f)
# plot_range = args["plot_range"]
''' load vars '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors'
os.makedirs(plot_dir_path, exist_ok=True)

''' calculate errors '''

def calculate_errors(realtime_predicts):
  prediction_errors = {} # prediction_errors[sensor][model][error_type] = [error values based on comm round]
  for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      prediction_errors[sensor_id] = {}

      for model, predicts in models_attr.items():
        if 'onestep' in model:
          prediction_errors[sensor_id][model] = {}
          prediction_errors[sensor_id][model]['MAE'] = []
          prediction_errors[sensor_id][model]['MSE'] = []
          prediction_errors[sensor_id][model]['RMSE'] = []
          prediction_errors[sensor_id][model]['MAPE'] = []

          for predict in predicts:
            round = predict[0]
            data = predict[1]
            true_data = models_attr['true'][round - 1][1]
            prediction_errors[sensor_id][model]['MAE'].append(get_MAE(true_data, data))
            prediction_errors[sensor_id][model]['MSE'].append(get_MSE(true_data, data))
            prediction_errors[sensor_id][model]['RMSE'].append(get_RMSE(true_data, data))
            prediction_errors[sensor_id][model]['MAPE'].append(get_MAPE(true_data, data))
  return prediction_errors

def plot_realtime_errors(prediction_errors, error_to_plot):
    for sensor_id, model_error in prediction_errors.items():
        print(f"Plotting {error_to_plot} error during real time FL simulation for {sensor_id}.")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(model_error['baseline_onestep'][error_to_plot]) + 1), model_error['baseline_onestep'][error_to_plot], label='baseline_onestep', color='#ffb839')
        ax.plot(range(1, len(model_error['global_onestep'][error_to_plot]) + 1), model_error['global_onestep'][error_to_plot], label='global_onestep', color='#5a773a')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.xlabel('Comm Round')
        plt.ylabel('Volume')
        plt.title(f'{sensor_id} - Real Rime {error_to_plot} Error')
        fig = plt.gcf()
        # fig.set_size_inches(228.5, 10.5)
        fig.set_size_inches(10.5, 3.5)
        print()
        plt.savefig(f'{plot_dir_path}/{sensor_id}_{error_to_plot}.png', bbox_inches='tight', dpi=100)
        plt.show()

prediction_errors = calculate_errors(realtime_predicts)
plot_realtime_errors(prediction_errors, args["error_type"])
