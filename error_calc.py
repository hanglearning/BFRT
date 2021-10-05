""" Error calculation functions """

# Prepare error calculation functions

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def get_MAE(y_true, y_pred):  
  return mean_absolute_error(y_true, y_pred)

def get_MSE(y_true, y_pred):  
  return mean_squared_error(y_true, y_pred, squared=True)

def get_RMSE(y_true, y_pred):  
  return mean_squared_error(y_true, y_pred, squared=False)

def get_MAPE(y_true, y_pred):  
  return mean_absolute_percentage_error(y_true, y_pred)