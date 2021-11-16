import argparse
import os
import pickle
import matplotlib.pyplot as plt

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg plot learning curves")

# arguments for system vars
parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the all_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-pl', '--plot_last_comm_rounds', type=int, default=24, help='the number of the last comm rounds to plot')

args = parser.parse_args()
args = args.__dict__

''' Variables Required '''
input_length = 12
logs_dirpath = args["logs_dirpath"]
plotting_range = args["plot_last_comm_rounds"] # to plot last plotting_range hours
''' Variables Required '''

with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
	config_vars = pickle.load(f)
input_length = config_vars["input_length"]

plot_dir_path = f'{logs_dirpath}/plots'
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
      # customize x axis labels
      my_xticks = []
      for i in range(len(x)):
        if i % input_length == 0:
          my_xticks.append(i//input_length + 1)
        else:
          my_xticks.append('')

      plt.xticks(x, my_xticks)

      ax.plot(x[-int(60/5*plotting_range):], plot_data['true']['y'][-int(60/5*plotting_range):], label='True Data')
      
      ax.plot(x[-int(60/5*plotting_range):], plot_data['global_chained']['y'][-int(60/5*plotting_range):], label='global_chained', color='darkgreen')
      ax.plot(x[-int(60/5*plotting_range):], plot_data['global_onestep']['y'][-int(60/5*plotting_range):], label='global_onestep', color='lime')
      ax.plot(x[-int(60/5*plotting_range):], plot_data['global_multi']['y'][-int(60/5*plotting_range):], label='global_multi', color='limegreen')

      ax.plot(x[-int(60/5*plotting_range):], plot_data['baseline_chained']['y'][-int(60/5*plotting_range):], label='baseline_chained', color='darkred')
      ax.plot(x[-int(60/5*plotting_range):], plot_data['baseline_onestep']['y'][-int(60/5*plotting_range):], label='baseline_onestep', color='red')
      ax.plot(x[-int(60/5*plotting_range):], plot_data['baseline_multi']['y'][-int(60/5*plotting_range):], label='baseline_multi', color='lightsalmon')

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

with open(f"{logs_dirpath}/all_predicts.pkl", 'rb') as f:
    sensor_predicts = pickle.load(f)
plot_and_save(sensor_predicts)