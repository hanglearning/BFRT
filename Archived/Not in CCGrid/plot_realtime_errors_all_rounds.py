# plot 4 errors for the entire FL simulation
# NOT used in CCGrid 22

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
parser.add_argument('-sr', '--starting_comm_round', type=int, default=0, help='epoch number to start plotting')
parser.add_argument('-er', '--ending_comm_round', type=int, default=None, help='epoch number to end plotting')
parser.add_argument('-ei', '--error_interval', type=int, default=1, help='unit is comm rounds')

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
s_round = args["starting_comm_round"]
e_round = args["ending_comm_round"]
''' load vars '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors_all_rounds'
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

          processed_rounds = set() # see plot_realtime_learning_curves.py, to be deleted in final version
          for predict in predicts:
            round = predict[0]
            if round not in processed_rounds:
              processed_rounds.add(round)
              data = predict[1]
              true_data = models_attr['true'][round - 1][1]
              prediction_errors[sensor_id][model]['MAE'].append(get_MAE(true_data, data))
              prediction_errors[sensor_id][model]['MSE'].append(get_MSE(true_data, data))
              prediction_errors[sensor_id][model]['RMSE'].append(get_RMSE(true_data, data))
              prediction_errors[sensor_id][model]['MAPE'].append(get_MAPE(true_data, data))
  return prediction_errors

def plot_realtime_errors(prediction_errors, error_to_plot, e_round):
    for sensor_id, model_error in prediction_errors.items():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.setp(ax, ylim=(0, 400))
        # if e_round is not specified
        if not e_round:
          e_round = min(len(model_error['baseline_onestep'][error_to_plot]), len(model_error['global_onestep'][error_to_plot]))
        print(f"Plotting {error_to_plot} error during real time FL simulation for {sensor_id} from round {s_round} to round {e_round}.")
        ax.plot(range(1, len(model_error['baseline_onestep'][error_to_plot]) + 1)[s_round:e_round+1], model_error['baseline_onestep'][error_to_plot][s_round:e_round+1], label='baseline_onestep', color='#ffb839')
        ax.plot(range(1, len(model_error['global_onestep'][error_to_plot]) + 1)[s_round:e_round+1], model_error['global_onestep'][error_to_plot][s_round:e_round+1], label='global_onestep', color='#5a773a')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.xlabel('Comm Round')
        plt.ylabel('Error')
        plt.title(f'{sensor_id} - Real Rime {error_to_plot} Error')
        fig = plt.gcf()
        # fig.set_size_inches(228.5, 10.5)
        fig.set_size_inches(10.5, 3.5)
        print()
        plt.savefig(f'{plot_dir_path}/{sensor_id}_{error_to_plot}.png', bbox_inches='tight', dpi=100)
        plt.show()

prediction_errors = calculate_errors(realtime_predicts)
plot_realtime_errors(prediction_errors, args["error_type"], e_round) # not sure why must pass e_round here

