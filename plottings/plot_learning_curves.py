import argparse
import os
import pickle
import matplotlib.pyplot as plt

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg plot learning curves")

# arguments for system vars
parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-pl', '--plot_last_comm_rounds', type=int, default=24, help='The number of the last comm rounds to plot. Will be a backup if starting_epoch and ending_epoch are not specified.')
parser.add_argument('-se', '--starting_epoch', type=int, default=None, help='epoch number to start plotting')
parser.add_argument('-ee', '--ending_epoch', type=int, default=None, help='epoch number to end plotting')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')


args = parser.parse_args()
args = args.__dict__

''' Variables Required '''
logs_dirpath = args["logs_dirpath"]
with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
input_length = config_vars["input_length"]
plot_last_comm_rounds = args["plot_last_comm_rounds"] # to plot last plot_last_comm_rounds hours
time_res = args["time_resolution"]
s_epoch = args["starting_epoch"]
e_epoch = args["ending_epoch"]
''' Variables Required '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_learning_curves'
os.makedirs(plot_dir_path, exist_ok=True)

def plot_and_save(sensor_predicts):
    """Plot
    Plot the true data and predicted data.
    Plot the errors between true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    for sensor_file, models_attr in sensor_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      plot_data = {}

      for model, predicts in models_attr.items():

        plot_data[model] = {}
        plot_data[model]['x'] = []
        plot_data[model]['y'] = []
        
        for predict in predicts:
          round = predict[0]
          data = predict[1]
          plot_data[model]['x'].extend(range((round - 1) * input_length + 1, round * input_length + 1))
          plot_data[model]['y'].extend(data)

      
      fig = plt.figure()
      ax = fig.add_subplot(111)
      
      x = plot_data['true']['x']
      # customize x axis labels as comm rounds
      my_xticks = []
      for x_ele in x:
        if (x_ele - 1) % input_length == 0:
          my_xticks.append(x_ele//input_length + 1)
        else:
          my_xticks.append('')

      plt.xticks(x, my_xticks)

      if s_epoch and e_epoch:
        print(f"Plotting comm round {s_epoch} to {e_epoch} for {sensor_id}.")
        start_range = int(60/time_res*(s_epoch-1))
        end_range = int(60/time_res*e_epoch)
        ax.plot(x[start_range:end_range], plot_data['true']['y'][start_range:end_range], label='True Data')
        
        # ax.plot(x[start_range:end_range], plot_data['global_chained']['y'][start_range:end_range], label='global_chained', color='darkgreen')
        ax.plot(x[start_range:end_range], plot_data['global_onestep']['y'][start_range:end_range], label='global_onestep', color='#5a773a')
        # ax.plot(x[start_range:end_range], plot_data['global_multi']['y'][start_range:end_range], label='global_multi', color='limegreen')

        # ax.plot(x[start_range:end_range], plot_data['baseline_chained']['y'][start_range:end_range], label='baseline_chained', color='darkred')
        ax.plot(x[start_range:end_range], plot_data['baseline_onestep']['y'][start_range:end_range], label='baseline_onestep', color='#ffb839')
        # ax.plot(x[start_range:end_range], plot_data['baseline_multi']['y'][start_range:end_range], label='baseline_multi', color='lightsalmon')
      else:
        # if plot_last_comm_rounds > 22, there is an empty comm at the begining for x axis
        print(f"Plotting last {plot_last_comm_rounds} comm rounds for {sensor_id}.")
        plotting_range = -int(60/time_res*plot_last_comm_rounds)
        ax.plot(x[plotting_range:], plot_data['true']['y'][plotting_range:], label='True Data')
        
        # ax.plot(x[plotting_range:], plot_data['global_chained']['y'][plotting_range:], label='global_chained', color='darkgreen')
        ax.plot(x[plotting_range:], plot_data['global_onestep']['y'][plotting_range:], label='global_onestep', color='#5a773a')
        # ax.plot(x[plotting_range:], plot_data['global_multi']['y'][plotting_range:], label='global_multi', color='limegreen')

        # ax.plot(x[plotting_range:], plot_data['baseline_chained']['y'][plotting_range:], label='baseline_chained', color='darkred')
        ax.plot(x[plotting_range:], plot_data['baseline_onestep']['y'][plotting_range:], label='baseline_onestep', color='#ffb839')
        # ax.plot(x[plotting_range:], plot_data['baseline_multi']['y'][plotting_range:], label='baseline_multi', color='lightsalmon')

      plt.legend()
      plt.grid(True)
      plt.xlabel('Comm Round')
      plt.ylabel('Volume')
      plt.title(sensor_id)
      fig = plt.gcf()
      # fig.set_size_inches(228.5, 10.5)
      fig.set_size_inches(10.5, 3.5)
      print()
      plt.savefig(f'{plot_dir_path}/{sensor_id}.png', bbox_inches='tight', dpi=100)
      plt.show()

with open(f"{logs_dirpath}/realtime_predicts.pkl", 'rb') as f:
    sensor_predicts = pickle.load(f)
plot_and_save(sensor_predicts)