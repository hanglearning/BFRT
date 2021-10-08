import argparse
import pickle

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

parser.add_argument('-ap', '--all_predicts_path', type=str, default=None, help='the path to the all predicts file')
parser.add_argument('-er', '--error_rounds', type=int, default=24, help='Show error values of the last -er number of comm rounds. For example, "-er 24" would calculate errors and show for last 24 comm rounds (1 day)')


args = parser.parse_args()
args = args.__dict__

# with open(predictions_record_saved_path, 'rb') as f:
with open(args['all_predicts_path'], 'rb') as f:
    sensor_predicts = pickle.load(f)

for sensor_file, models_attr in sensor_predicts.items():
    
    sensor_id = sensor_file.split('.')[0]
    
    # record true data
    true_data = {}
    predicts = models_attr['true']
    for predict in predicts[-args['error_rounds']:]:
        round = predict[0]
        data = predict[1]
        true_data[round] = data

    # record global data
    global_data = {}
    predicts = models_attr['global']
    for predict in predicts[-args['error_rounds']:]:
        round = predict[0]
        data = predict[1]
        global_data[round] = data

    error_values = {}
    # calculate and record errors
    for round, data in true_data.items():
        error_values[round] = {}
        error_values[round]['MAE'] = get_MAE(data, global_data[round])
        error_values[round]['MSE'] = get_MSE(data, global_data[round])
        error_values[round]['RMSE'] = get_RMSE(data, global_data[round])
        error_values[round]['MAPE'] = get_MAPE(data, global_data[round])

print(error_values)