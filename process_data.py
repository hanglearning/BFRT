import numpy as np

""" Functions to process the data before training, including constructing the training and test data with scaling

NOTE - set the `attr` value for the learning and predicting feature (consistent in both functions)
"""

# Prepare Data Preprocessing function
"""
Processing the data
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler

attr = 'volume'

def process_pretrain_data(df_train, INPUT_LENGTH):
    """Process data

    # Arguments
        df_train: pandas data frame, trainig data.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
    """

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train[attr].values.reshape(-1, 1))
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train_set, test_set = [], []
    for i in range(INPUT_LENGTH, len(flow_train)):
        train_set.append(flow_train[i - INPUT_LENGTH: i + 1])

    train = np.array(train_set)
    
    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train


def process_pretrain_data_multi(df_train, INPUT_LENGTH, num_last_layer_neurons):
    """Process data

    # Arguments
        df_train: pandas data frame, trainig data.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
    """

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train[attr].values.reshape(-1, 1))
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train_set, test_set = [], []
    for i in range(INPUT_LENGTH, len(flow_train) - (num_last_layer_neurons - 1)):
        train_set.append(flow_train[i - INPUT_LENGTH: i + num_last_layer_neurons])

    train = np.array(train_set)
    
    X_train = train[:, :-num_last_layer_neurons]
    y_train = train[:, -num_last_layer_neurons:]

    return X_train, y_train

def get_scaler(df_whole):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_whole[attr].values.reshape(-1, 1))
    return scaler

def process_train_data_single(df_train, scaler, INPUT_LENGTH):
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    train_set = []
    for i in range(INPUT_LENGTH, len(flow_train)):
        train_set.append(flow_train[i - INPUT_LENGTH: i + 1])
    train = np.array(train_set)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train

def process_train_data_multi(df_train, scaler, INPUT_LENGTH, num_last_layer_neurons):
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    train_set = []
    for i in range(INPUT_LENGTH, len(flow_train) - (num_last_layer_neurons - 1)):
        train_set.append(flow_train[i - INPUT_LENGTH: i + num_last_layer_neurons])

    train = np.array(train_set)
    
    X_train = train[:, :-num_last_layer_neurons]
    y_train = train[:, -num_last_layer_neurons:]

    return X_train, y_train

def process_test_one_step(df_test, scaler, INPUT_LENGTH):
    flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    test_set = []
    for i in range(INPUT_LENGTH, len(flow_test)):
        test_set.append(flow_test[i - INPUT_LENGTH: i + 1])
    test = np.array(test_set) 
    X_test = test[:, :-1]
    y_test = test[:, -1]
    return X_test, y_test

def process_test_chained(df_test, scaler, INPUT_LENGTH):
    flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    test_set = [flow_test]
    # for i in range(INPUT_LENGTH, len(flow_test)):
    #     test_set.append(flow_test[i - INPUT_LENGTH: i + 1])
    # test = np.array(test_set) 
    # X_test = test[:, :-1]
    X_test= np.array(test_set) 
    return X_test

def process_test_multi_and_get_y_true(df_test, scaler, INPUT_LENGTH, num_last_layer_neurons):
    
    flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    test_set = []
    
    for i in range(INPUT_LENGTH, len(flow_test) - (num_last_layer_neurons - 1)):
        test_set.append(flow_test[i - INPUT_LENGTH: i + num_last_layer_neurons])
    test = np.array(test_set) 
    
    X_test = test[:, :-num_last_layer_neurons]
    y_true = test[:, -num_last_layer_neurons:]

    return X_test, y_true 

