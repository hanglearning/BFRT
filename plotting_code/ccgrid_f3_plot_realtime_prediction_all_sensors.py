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
parser.add_argument('-v', '--version', type=str, default='c', help='c for ccgrid for one step model, or j for journal extension for chained or multi-output models')
parser.add_argument('-r', '--representative', type=str, default=None, help='detector id to be the representative figure. By default, it is kind of random')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in subplots')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in subplots')

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
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    sys.exit(f"Please specify the number of columns.")
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
  
def plot_and_save_two_rows(sensor_lists, plot_data, COL):
    """Plot
    Plot the true data and predicted data.
    """
    
    # draw 1 plot
    # for sensor_id in sensor_lists:
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.setp(ax, ylim=(0, 800))
    fig.text(0.04, 0.5, 'Volume', va='center', rotation='vertical')
    ax.set_xlabel('Round Index')
    # sensor_id = sensor_lists[0]
    sensor_id = args['representative']
    
    ax.set_title(sensor_id)
    
    
    plotting_range = int(60/time_res*plot_last_comm_rounds)
    my_xticks = deque(range((sing_x_density - 1) * input_length, plotting_range, input_length * sing_x_density))
    my_xticks.appendleft(0)
    
    ax.set_xticks(my_xticks)
    xticklabels = list(range(config_vars["last_round"] - plot_last_comm_rounds, config_vars["last_round"] + 1, sing_x_density))
    # xticklabels[0] = 1
    ax.set_xticklabels(xticklabels, Fontsize = 9, rotation = 45)
    
    ax.plot(range(plotting_range), plot_data[sensor_id]['true']['y'][-plotting_range:], label='True Data', color='blue')
    true_curve = mlines.Line2D([], [], color='blue', label="TRUE")
    
    if args['version'] == 'c':
        
        ''' CCGrid Version '''
    
        ax.plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

        ax.plot(range(plotting_range), plot_data[sensor_id]['baseline_onestep']['y'][-plotting_range:], label='baseline_onestep', color='orange')

        baseline_curve = mlines.Line2D([], [], color='orange', label="BASE")
        global_curve = mlines.Line2D([], [], color='lime', label="FED")
        
        ax.legend(handles=[true_curve,baseline_curve, global_curve], loc='best', prop={'size': 10})
        fig.set_size_inches(8, 2)
        plt.savefig(f'{plot_dir_path}/single_figure.png', bbox_inches='tight', dpi=500)
        # plt.show()
    
    else:    
        ''' Journal Extension '''
        
        ax.plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

        ax.plot(range(plotting_range), plot_data[sensor_id]['global_chained']['y'][-plotting_range:], label='global_chained', color='orange')
        
        ax.plot(range(plotting_range), plot_data[sensor_id]['global_multi']['y'][-plotting_range:], label='global_multi', color='darkgreen')

        global_onestep_curve = mlines.Line2D([], [], color='lime', label="ONE")
        global_chained_curve = mlines.Line2D([], [], color='orange', label="CHAINED")
        global_multi_curve = mlines.Line2D([], [], color='darkgreen', label="MULTI")
        
        ax.legend(handles=[true_curve,global_onestep_curve, global_chained_curve, global_multi_curve], loc='best', prop={'size': 10})
        fig.set_size_inches(8, 2)
        plt.savefig(f'{plot_dir_path}/single_figure.png', bbox_inches='tight', dpi=500)
        # plt.show()
    
    # draw subplots
    # default - draw 2 row and 3 col = 6 plots
    # if COL == 1:
    #     fig, axs = plt.subplots(ROW, sharex=True, sharey=True)
    # else:
    if ROW == 1 and COL is None:
        COL = len(sensor_lists) - 1
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, 800))
    # axs[0].set_ylabel('Volume')
    # fig.text(0.5, 0.04, 'Round Index', ha='center', size=13)
    fig.text(0.04, 0.5, 'Volume', va='center', rotation='vertical', size=13)
    if ROW == 1 and COL == 1:
        axs.set_xlabel('Round Index', size=13)
    elif ROW == 1 and COL > 1:
        axs[COL//2].set_xlabel('Round Index', size=13)
    elif ROW > 1 and COL > 1:
        axs[ROW-1][COL//2].set_xlabel('Round Index', size=13)
    
    
    my_xticks = [0, plotting_range//2, plotting_range-2]
    my_xticklabels = [config_vars["last_round"] - plot_last_comm_rounds + 1, config_vars["last_round"] - plot_last_comm_rounds//2 + 1, config_vars["last_round"]]
    
    
    sensor_lists.remove(args['representative'])
    for sensor_plot_iter in range(len(sensor_lists)):
        # for 6 plots
        # sensor_plot_iter 0 ~ 2 -> row 0, col 0 1 2
        # sensor_plot_iter 3 ~ 5 ->row 1, col 0 1 2
        # row = sensor_plot_iter // 3
        # col = sensor_plot_iter % 3
        row = sensor_plot_iter // COL
        col = sensor_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplots = axs
        elif ROW == 1 and COL > 1:
          subplots = axs[sensor_plot_iter]
        elif ROW > 1 and COL > 1:
          subplots = axs[row][col]
        
        sensor_id = sensor_lists[sensor_plot_iter]
        # subplots.set_xlabel('Comm Round')
        subplots.set_title(sensor_id)
        
        plotting_range = int(60/time_res*plot_last_comm_rounds)
        
        subplots.set_xticks(my_xticks)
        subplots.set_xticklabels(my_xticklabels)
        
        subplots.plot(range(plotting_range), plot_data[sensor_id]['true']['y'][-plotting_range:], label='True Data', color='blue')
        true_curve = mlines.Line2D([], [], color='blue', label="TRUE")
        
        if args['version'] == 'c':
            ''' CCGrid Version '''
            subplots.plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

            subplots.plot(range(plotting_range), plot_data[sensor_id]['baseline_onestep']['y'][-plotting_range:], label='baseline_onestep', color='orange')
        
            baseline_curve = mlines.Line2D([], [], color='orange', label="BASE")
            global_curve = mlines.Line2D([], [], color='lime', label="FED")
            
            subplots.legend(handles=[true_curve,baseline_curve, global_curve], loc='best', prop={'size': 10})
            
        else:        
            ''' Journal Extension '''
            
            subplots.plot(range(plotting_range), plot_data[sensor_id]['global_onestep']['y'][-plotting_range:], label='global_onestep', color='lime')

            subplots.plot(range(plotting_range), plot_data[sensor_id]['global_chained']['y'][-plotting_range:], label='global_chained', color='orange')
            
            subplots.plot(range(plotting_range), plot_data[sensor_id]['global_multi']['y'][-plotting_range:], label='global_multi', color='darkgreen')
        
            global_onestep_curve = mlines.Line2D([], [], color='lime', label="ONE")
            global_chained_curve = mlines.Line2D([], [], color='orange', label="CHAINED")
            global_multi_curve = mlines.Line2D([], [], color='darkgreen', label="MULTI")
            
            subplots.legend(handles=[true_curve,global_onestep_curve, global_chained_curve, global_multi_curve], loc='best', prop={'size': 10})
            
        
        
    fig.set_size_inches(10, 4)
    plt.savefig(f'{plot_dir_path}/multi_figure.png', bbox_inches='tight', dpi=300)
    # plt.show()
    
def calculate_errors(plot_data):
    error_values = {}
    plotting_range = int(60/time_res*plot_last_comm_rounds)

    for sensor_id, prediction_method in plot_data.items():
        error_values[sensor_id] = {}
        for model, predicts in prediction_method.items():
            if model != 'true' and "global" in model:
                error_values[sensor_id][model] = {}
                error_values[sensor_id][model]['MAE'] = get_MAE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['MSE'] = get_MSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['RMSE'] = get_RMSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[sensor_id][model]['MAPE'] = get_MAPE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
    with open(f'{plot_dir_path}/errors.txt', "w") as file:
        for sensor_id, model in error_values.items():
            file.write(f'\nfor {sensor_id}\n')
            error_values_df = pd.DataFrame.from_dict(model)
            file.write(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
            file.write('\n')
    
with open(f"{logs_dirpath}/realtime_predicts.pkl", 'rb') as f:
    sensor_predicts = pickle.load(f)
sensor_lists, plot_data = make_plot_data(sensor_predicts)
# plot_and_save(sensor_lists, plot_data)
plot_and_save_two_rows(sensor_lists, plot_data, COL)
calculate_errors(plot_data)