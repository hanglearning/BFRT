# plot 4 errors during FL simulation

import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pandas as pd

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

from tabulate import tabulate

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="inferencing code")

parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-ei', '--error_interval', type=int, default=100, help='unit is comm rounds, used in showing error table')

args = parser.parse_args()
args = args.__dict__

''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
all_sensor_files = config_vars["all_sensor_files"]

with open(f'{logs_dirpath}/realtime_predicts.pkl', 'rb') as f:
    realtime_predicts = pickle.load(f)

''' load vars '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors_interval'
os.makedirs(plot_dir_path, exist_ok=True)

def construct_realtime_error_table(realtime_predicts):
    realtime_error_table = {}
    for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      realtime_error_table[sensor_id] = {}

      for model, predicts in models_attr.items():
        if 'onestep' in model:
          realtime_error_table[sensor_id][model] = {}
          realtime_error_table[sensor_id][model]['MAE'] = []
          realtime_error_table[sensor_id][model]['MSE'] = []
          realtime_error_table[sensor_id][model]['RMSE'] = []
          realtime_error_table[sensor_id][model]['MAPE'] = []

          processed_rounds = set() # see plot_realtime_learning_curves.py, to be deleted in final version
          data_list = []
          true_data_list = []
          for predict in predicts:
            round = predict[0]
            if round not in processed_rounds:
              processed_rounds.add(round)
              data = predict[1]
              true_data = models_attr['true'][round - 1][1]
              data_list.extend(data)
              true_data_list.extend(true_data)
              if round != 1 and (round - 1) % args["error_interval"] == 0:
                # conclude the errors
                realtime_error_table[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
                realtime_error_table[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
                realtime_error_table[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
                realtime_error_table[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
                data_list = []
                true_data_list = []
          # if there's leftover
          if data_list and true_data_list:
            realtime_error_table[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
            realtime_error_table[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
            realtime_error_table[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
            realtime_error_table[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
    return realtime_error_table
  

def plot_realtime_errors(prediction_errors, error_to_plot):
    for sensor_id, model_error in prediction_errors.items():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(f"Plotting {error_to_plot} error during real time FL simulation for {sensor_id} with interval {args['error_interval']}.")
        ax.plot(range(1, len(model_error['baseline_onestep'][error_to_plot]) + 1), model_error['baseline_onestep'][error_to_plot], label='baseline_onestep', color='#ffb839')
        ax.plot(range(1, len(model_error['global_onestep'][error_to_plot]) + 1), model_error['global_onestep'][error_to_plot], label='global_onestep', color='#5a773a')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.xlabel(f"Comm Round with Interval {args['error_interval']}")
        plt.ylabel('Error')
        plt.title(f'{sensor_id} - Real Rime {error_to_plot} Error')
        fig = plt.gcf()
        # fig.set_size_inches(228.5, 10.5)
        fig.set_size_inches(10.5, 3.5)
        print()
        plt.savefig(f"{plot_dir_path}/{sensor_id}_{error_to_plot}_interval_{args['error_interval']}.png", bbox_inches='tight', dpi=100)
        plt.show()

realtime_error_table = construct_realtime_error_table(realtime_predicts)

# show table
for sensor_id, model in realtime_error_table.items():
    print(f'\nfor {sensor_id}')
    error_values_df = pd.DataFrame.from_dict(model)
    print(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
    
# show plots
plot_realtime_errors(realtime_error_table, "MAE")

