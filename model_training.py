""" Model training functions for

(1) baseline model - a model that is trained solely by the device's own data without federation, simulating a centralized learning. Used to compare with federated models.

(2) local model - the local model defined in FL.
"""

# Prepare for TF training function
"""
Train the NN model.
"""

from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")
from keras.models import load_model
from build_lstm import build_lstm

def train_baseline_model(comm_round, model_path, X_train, y_train, sensor_id, this_sensor_dir_path, config):
    """train the baseline model 

    # Arguments
        comm_round: FL communication round number
        model_path: model weights to load to continue training
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        sensor_id: id of the sensor, e.g., 19985_NB
        this_sensor_dir_path: specify directory to store related records for this sensor
        config: Dict, parameter for train.
    """
    model = load_model(model_path)
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.00)
    
    # save model weights
    model_file_path = f'{this_sensor_dir_path}/h5/baseline/{sensor_id}_baseline_{comm_round}.h5'
    model.save(model_file_path)
    # Not significant to show loss in this study
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{this_sensor_dir_path}/loss/baseline/{sensor_id}_baseline_{comm_round}.csv', encoding='utf-8', index=False)
    return model_file_path


def train_local_model(comm_round, global_weights, X_train, y_train, sensor_id, this_sensor_dir_path, config, INPUT_LENGTH):
    """train the local model 

    # Arguments
        comm_round: FL communication round number
        global_weights: ndarray, global model weights to load to continue training
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        sensor_id: id of the sensor, e.g., 19985_NB
        this_sensor_dir_path: specify directory to store related records for this sensor
        config: Dict, parameter for train.
    """
    model = build_lstm([INPUT_LENGTH, 64, 64, 1])
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    model.set_weights(global_weights)
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.00)
    
    # save model weights
    model_file_path = f'{this_sensor_dir_path}/h5/local/{sensor_id}_local_{comm_round}.h5'
    model.save(model_file_path)
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{this_sensor_dir_path}/loss/local/{sensor_id}_local_{comm_round}.csv', encoding='utf-8', index=False)
    return model_file_path, model.get_weights()