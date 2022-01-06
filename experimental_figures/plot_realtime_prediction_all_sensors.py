import argparse
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import deque

import pandas as pd
from tabulate import tabulate

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

# ''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg plot learning curves")

# arguments for system vars
parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-pl', '--plot_last_comm_rounds', type=int, default=24, help='The number of the last comm rounds to plot. Will be a backup if starting_comm_round and ending_comm_round are not specified.')
parser.add_argument('-sr', '--starting_comm_round', type=int, default=None, help='epoch number to start plotting')
parser.add_argument('-er', '--ending_comm_round', type=int, default=None, help='epoch number to end plotting')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')
parser.add_argument('-sd', '--single_plot_x_axis_density', type=int, default=1, help='label the 1 large plot x-axis by every this number of ticks')
parser.add_argument('-md', '--multi_plot_x_axis_density', type=int, default=4, help='label the multi sub plots x-axis by every this number of ticks')

args = parser.parse_args()
args = args.__dict__

''' Variables Required '''
logs_dirpath = args["logs_dirpath"]
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
input_length = config_vars["input_length"]
plot_last_comm_rounds = args["plot_last_comm_rounds"] # to plot last plot_last_comm_rounds hours
time_res = args["time_resolution"]
s_round = args["starting_comm_round"]
e_round = args["ending_comm_round"]
sing_x_density = args["single_plot_x_axis_density"]
mul_x_density = args["multi_plot_x_axis_density"]
''' Variables Required '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_learning_curves_all_sensors'
os.makedirs(plot_dir_path, exist_ok=True)

def make_plot_data(sensor_predicts):
    sensor_lists = [sensor_file.split('.')[0] for sensor_file in sensor_predicts.keys()]
    
    plot_data = {}
    for sensor_file, models_attr in sensor_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      plot_data[sensor_id] = {}

      for model, predicts in models_attr.items():

        plot_data[sensor_id][model] = {}
        plot_data[sensor_id][model]['x'] = []
        plot_data[sensor_id][model]['y'] = []
        
        processed_rounds = set()
        for predict in predicts:
          round = predict[0]
          if round not in processed_rounds:
            # a simple hack to be backward compatible to the sensor_predicts in main.py which may contain duplicate training round due to resuming, to be deleted in final version
            processed_rounds.add(round)
            data = predict[1]
            plot_data[sensor_id][model]['x'].extend(range((round - 1) * input_length + 1, round * input_length + 1))
            plot_data[sensor_id][model]['y'].extend(data)

    return sensor_lists, plot_data
  
def plot_and_save_two_rows(sensor_lists, plot_data):
    """Plot
    Plot the true data and predicted data.
    Plot the errors between true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    
    # draw 1 plot
    # for sensor_id in sensor_lists:
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.setp(ax, ylim=(0, 800))
    fig.text(0.04, 0.5, 'Volume', va='center', rotation='vertical')
    ax.set_xlabel('Round Index')
    sensor_id = sensor_lists[0]
    
    ax.set_title(sensor_id)
    
    
    plotting_range = int(60/time_res*plot_last_comm_rounds)
    my_xticks = deque(range((sing_x_density - 1) * input_length, plotting_range, input_length * sing_x_density))
    my_xticks.appendleft(0)
    
    ax.set_xticks(my_xticks)
    xticklabels = list(range(config_vars["last_round"] - plot_last_comm_rounds, config_vars["last_round"] + 1, sing_x_density))
    # xticklabels[0] = 1
    ax.set_xticklabels(xticklabels, Fontsize = 9, rotation = 45)
    ax.plot(range(plotting_range), plot_data[sensor_id]['true']['y'][-plotting_range:], label='True Data', color='blue')
    
    ax.plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

    ax.plot(range(plotting_range), plot_data[sensor_id]['baseline_onestep']['y'][-plotting_range:], label='baseline_onestep', color='orange')

    true_curve = mlines.Line2D([], [], color='blue', label="TRUE")
    baseline_curve = mlines.Line2D([], [], color='orange', label="BASE")
    global_curve = mlines.Line2D([], [], color='lime', label="FED")
    
    ax.legend(handles=[true_curve,baseline_curve, global_curve], loc='best', prop={'size': 10})
    fig.set_size_inches(8, 2)
    plt.savefig(f'{plot_dir_path}/single_figure.png', bbox_inches='tight', dpi=500)
    # plt.show()
    
    # draw 2 row 6 plots
    
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, 800))
    # axs[0].set_ylabel('Volume')
    # fig.text(0.5, 0.04, 'Round Index', ha='center', size=13)
    fig.text(0.04, 0.5, 'Volume', va='center', rotation='vertical', size=13)
    axs[1][1].set_xlabel('Round Index', size=13)
    
    # HARD CODE
    my_xticks = [0, plotting_range//2, plotting_range-2]
    my_xticklabels = [1142, 1154, 1165]
    
      
    for sensor_plot_iter in range(len(sensor_lists[1:])):
        # sensor_plot_iter 0 ~ 2 -> row 0, col 0 1 2
        # sensor_plot_iter 3 ~ 5 ->row 1, col 0 1 2
        row = sensor_plot_iter // 3
        col = sensor_plot_iter % 3
        
        sensor_id = sensor_lists[1:][sensor_plot_iter]
        # axs[row][col].set_xlabel('Comm Round')
        axs[row][col].set_title(sensor_id)
        
        plotting_range = int(60/time_res*plot_last_comm_rounds)
        
        axs[row][col].set_xticks(my_xticks)
        axs[row][col].set_xticklabels(my_xticklabels)
        
        axs[row][col].plot(range(plotting_range), plot_data[sensor_id]['true']['y'][-plotting_range:], label='True Data', color='blue')
        
        axs[row][col].plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

        axs[row][col].plot(range(plotting_range), plot_data[sensor_id]['baseline_onestep']['y'][-plotting_range:], label='baseline_onestep', color='orange')
    
        true_curve = mlines.Line2D([], [], color='blue', label="TRUE")
        baseline_curve = mlines.Line2D([], [], color='orange', label="BASE")
        global_curve = mlines.Line2D([], [], color='lime', label="FED")
        
        axs[row][col].legend(handles=[true_curve,baseline_curve, global_curve], loc='best', prop={'size': 10})
    fig.set_size_inches(10, 4)
    plt.savefig(f'{plot_dir_path}/multi_figure.png', bbox_inches='tight', dpi=300)
    # plt.show()
    
def calculate_errors(plot_data):
    error_values = {}
    plotting_range = int(60/time_res*plot_last_comm_rounds)

    for sensor_id, prediction_method in plot_data.items():
        error_values[sensor_id] = {}
        for model, predicts in prediction_method.items():
            if model != 'true' and "onestep" in model:
                error_values[sensor_id][model] = {}
                error_values[sensor_id][model]['MAE'] = get_MAE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['MSE'] = get_MSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['RMSE'] = get_RMSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['MAPE'] = get_MAPE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
    for sensor_id, model in error_values.items():
        print(f'\nfor {sensor_id}')
        error_values_df = pd.DataFrame.from_dict(model)
        print(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
    
with open(f"{logs_dirpath}/realtime_predicts.pkl", 'rb') as f:
    sensor_predicts = pickle.load(f)
sensor_lists, plot_data = make_plot_data(sensor_predicts)
# plot_and_save(sensor_lists, plot_data)
plot_and_save_two_rows(sensor_lists, plot_data)
calculate_errors(plot_data)