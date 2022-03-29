# plot 4 errors during FL simulation with a specified interval
# Figure 4 in CCGrid

import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

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
parser.add_argument('-et', '--error_type', type=str, default="MAE", help='the error type to plot and calculate')
parser.add_argument('-yt', '--y_top', type=int, default=150, help='the max error value on y-axis')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in the plot')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in the plot')

args = parser.parse_args()
args = args.__dict__

'''
Note that if there was X rounds in FL, the real time predictions were (X-1) rounds rather than X rounds. The first round's prediction will predict for 2nd round, and so.
'''

''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
all_sensor_files = config_vars["all_sensor_files"]

with open(f'{logs_dirpath}/realtime_predicts.pkl', 'rb') as f:
    realtime_predicts = pickle.load(f)
    
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    sys.exit(f"Please specify the number of columns.")
''' load vars '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors_interval'
os.makedirs(plot_dir_path, exist_ok=True)

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

def construct_realtime_error_table(realtime_predicts):
    realtime_error_table_normalized = {}
    for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      realtime_error_table_normalized[sensor_id] = {}
      
      for model, predicts in models_attr.items():
        if 'onestep' in model:
          realtime_error_table_normalized[sensor_id][model] = {}
          realtime_error_table_normalized[sensor_id][model]['MAE'] = []
          realtime_error_table_normalized[sensor_id][model]['MSE'] = []
          realtime_error_table_normalized[sensor_id][model]['RMSE'] = []
          realtime_error_table_normalized[sensor_id][model]['MAPE'] = []

          processed_rounds = set() # see plot_realtime_learning_curves.py, to be deleted in final version
          data_list = []
          true_data_list = []
          for predict_iter in range(len(predicts)):
            predict = predicts[predict_iter] 
            # see plot_realtime_learning_curves.py, to be deleted in final version
            round = predict[0]
            if round not in processed_rounds:
              processed_rounds.add(round)
              data = predict[1]
              true_data = models_attr['true'][predict_iter][1]
              data_list.extend(data)
              true_data_list.extend(true_data)
              if round != 1 and (round - 1) % args["error_interval"] == 0:
                # conclude the errors
                realtime_error_table_normalized[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
                data_list = []
                true_data_list = []
          # if there's leftover
          if data_list and true_data_list:
            realtime_error_table_normalized[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
    return realtime_error_table_normalized
  

def plot_realtime_errors_one_by_one(prediction_errors, error_to_plot):
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

def compare_l1_smaller_equal_percent(l1, l2):
  if len(l1) != len(l2):
    return "Error"
  l1_smaller_or_equal_values_count = 0
  for _ in range(len(l1)):
    if l1[_] < l2[_]:
      l1_smaller_or_equal_values_count += 1
  percentage = l1_smaller_or_equal_values_count/len(l1)
  percent_string = f"{percentage:.2%}"
  return percentage, percent_string

def plot_realtime_errors_all_sensors(realtime_error_table, all_prediction_errors, error_to_plot, ylim, COL):
    sensor_lists = [sensor_file.split('.')[0] for sensor_file in all_sensor_files]
    # sensor_lists = ['19940_03-10-2017_to_05-10-2017']
    # if ROW == 1:
    #   COL = len(sensor_lists)
    if ROW == 1 and COL is None:
        COL = len(sensor_lists)
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, ylim))
    if ROW == 1 and COL == 1:
        axs.set_xlabel('Round Range', size=13)
        axs.set_ylabel(f'{error_to_plot} Error', size=13)
    elif ROW == 1 and COL > 1:
        axs[COL//2].set_xlabel('Round Range', size=13)
        axs[0].set_ylabel(f'{error_to_plot} Error', size=13)
    elif ROW > 1 and COL > 1:
        axs[ROW-1][COL//2].set_xlabel('Round Range', size=13)
        axs[ROW//2][0].set_ylabel(f'{error_to_plot} Error', size=13)
      
    # axs[0].set_ylabel(f'{error_to_plot} Error', size=13)
    # fig.text(0.5, 0.04, 'Comm Round Index', ha='center', size=13)
    # axs[COL//2].set_xlabel(f'Round Range', size=13)
    
    for sensor_plot_iter in range(len(sensor_lists)):
      
        row = sensor_plot_iter // COL
        col = sensor_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplots = axs
        elif ROW == 1 and COL > 1:
          subplots = axs[sensor_plot_iter]
        elif ROW > 1 and COL > 1:
          subplots = axs[row][col]
          
        sensor_id = sensor_lists[sensor_plot_iter]
        model_error_normalized = realtime_error_table[sensor_id]
        
        subplots.set_title(sensor_id)
        
        num_of_plot_points = len(model_error_normalized['baseline_onestep'][error_to_plot])
        e_interval = args["error_interval"]
        
        
        subplots.set_xticks([0, num_of_plot_points // 2, num_of_plot_points - 1])
        subplots.set_xticklabels([f'  1-\n{e_interval}', f'{(num_of_plot_points // 2 - 1) * e_interval}-\n{num_of_plot_points // 2 * e_interval}', f'{(num_of_plot_points - 1) * e_interval}-\n{config_vars["last_round"]}'], fontsize=8)
        
        # compare two models and show smaller-error-value percentage
        # normalized
        global_better_percent_val, global_better_percent_string = compare_l1_smaller_equal_percent(model_error_normalized['global_onestep'][error_to_plot], model_error_normalized['baseline_onestep'][error_to_plot])
        
        # all real
        # global_better_percent_val, global_better_percent_string = compare_l1_smaller_equal_percent(all_prediction_errors[sensor_id]['global_onestep'][error_to_plot], all_prediction_errors[sensor_id]['baseline_onestep'][error_to_plot])
        
        subplots.plot(range(0, num_of_plot_points), model_error_normalized['baseline_onestep'][error_to_plot], label='baseline_onestep', color='#ffb839')
        
        if global_better_percent_val >= 0.5:
          annotation_color = 'red'
        else:
          annotation_color = 'black'
        subplots.annotate(global_better_percent_string, xy=(num_of_plot_points - 5, model_error_normalized['baseline_onestep'][error_to_plot][-5] + 10), size=8, color=annotation_color)
        
        subplots.plot(range(len(model_error_normalized['global_onestep'][error_to_plot])), model_error_normalized['global_onestep'][error_to_plot], label='global_onestep', color='#5a773a')
        
        if sensor_plot_iter == 0:   
          baseline_curve = mlines.Line2D([], [], color='#ffb839', label="BASE")
          global_curve = mlines.Line2D([], [], color='#5a773a', label="FED")
        
          subplots.legend(handles=[baseline_curve, global_curve], loc='best', prop={'size': 10})
    # show legend on each plot
    # baseline_curve = mlines.Line2D([], [], color='#ffb839', label="BASE")
    # global_curve = mlines.Line2D([], [], color='#5a773a', label="FED")    
    # fig.legend(handles=[baseline_curve, global_curve], loc='upper center')
    fig.set_size_inches(10, 1.3)
    plt.savefig(f'{plot_dir_path}/real_time_errors_all_sensors_{error_to_plot}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    
realtime_error_table = construct_realtime_error_table(realtime_predicts)

# show table
with open(f'{plot_dir_path}/errors.txt', "w") as file:
  for sensor_id, model in realtime_error_table.items():  
      file.write(f'\nfor {sensor_id}')
      error_values_df = pd.DataFrame.from_dict(model)
      file.write(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
      file.write('\n')
    
# show plots
all_prediction_errors = calculate_errors(realtime_predicts) # calculate global model outperform percentage
plot_realtime_errors_all_sensors(realtime_error_table, all_prediction_errors, args['error_type'], args['y_top'], COL)
# error_ylim = dict(MAE = 200, MSE = 6000, RMSE = 80, MAPE = 0.6)
# for error_type in error_ylim.keys():
#   plot_realtime_errors_all_sensors(realtime_error_table, error_type, error_ylim[error_type])
