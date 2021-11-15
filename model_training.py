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
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from build_lstm import build_lstm
from build_gru import build_gru

def train_baseline_model(model, comm_round, X_train, y_train, sensor_id, this_sensor_dirpath, single_or_multi, batch, epochs):
    """train the baseline model 

    # Arguments
        comm_round: FL communication round number
        model_path: model weights to load to continue training
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        sensor_id: id of the sensor, e.g., 19985_NB
        this_sensor_dirpath: specify directory to store related records for this sensor
        config: Dict, parameter for train.
    """
    #model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.00)
    
    # save model weights
    model_file_path = f'{this_sensor_dirpath}/baseline_{single_or_multi}_h5/comm_{comm_round}.h5'
    model.save(model_file_path)
    # Not significant to show loss in this study
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{this_sensor_dirpath}/loss/baseline/{sensor_id}_baseline_{comm_round}.csv', encoding='utf-8', index=False)
    return model_file_path


def train_local_model(model, comm_round, X_train, y_train, sensor_id, this_sensor_dirpath, single_or_multi, batch, epochs):
    """train the local model 

    # Arguments
        comm_round: FL communication round number
        global_weights: ndarray, global model weights to load to continue training
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        sensor_id: id of the sensor, e.g., 19985_NB
        this_sensor_dirpath: specify directory to store related records for this sensor
        config: Dict, parameter for train.
    """
    hist = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.00)
    
    # save model weights
    model_file_path = f'{this_sensor_dirpath}/local_{single_or_multi}_h5/comm_{comm_round}.h5'
    model.save(model_file_path)
    # loss_df = pd.DataFrame.from_dict(hist.history)
    # loss_df.to_csv(f'{this_sensor_dirpath}/loss/local/{sensor_id}_local_{comm_round}.csv', encoding='utf-8', index=False)
    return model_file_path, model.get_weights()