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


def process_data(df_train, df_test, is_chained, INPUT_LENGTH):
    """Process data
    # Arguments
        df_train: pandas data frame, trainig data.
        df_test: pandas data frame, test data.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        # X_test: ndarray. NOT NEEDED IN CHAINED PREDICTION
        y_test: ndarray.
        scaler: StandardScaler.
    """

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train[attr].values.reshape(-1, 1))
    X_train, y_train, X_test, y_test = None, None, None, None

    if df_train:
        flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
        train_set = []
        for i in range(INPUT_LENGTH, len(flow_train)):
            train_set.append(flow_train[i - INPUT_LENGTH: i + 1])
        train = np.array(train_set)
        X_train = train[:, :-1]
        y_train = train[:, -1]
    
    if df_test:
        flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
        test_set = []
        if is_chained:
            test_set.append(flow_test)
        else:
            for i in range(INPUT_LENGTH, len(flow_test)):
                test_set.append(flow_test[i - INPUT_LENGTH: i + 1])
        test = np.array(test_set) 
        if is_chained:
            X_test = None
            y_test = test.flatten()
        else:
            X_test = test[:, :-1]
            y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler
