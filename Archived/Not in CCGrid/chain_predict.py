"""Sliding INPUT_LENGTH data points per window."""
import numpy as np

# code for chain prediction
def chain_predict(model, initial_prediction_set, INPUT_LENGTH):

  initial_prediction_set = np.reshape(initial_prediction_set, (initial_prediction_set.shape[0], initial_prediction_set.shape[1], 1))
  chained_predictions = model.predict(initial_prediction_set)
  chained_predictions = chained_predictions.reshape(chained_predictions.shape + (1,))
  
  for start_index in range(1, INPUT_LENGTH):
    next_INPUT_LENGTH_minus_ONE_points_slice = initial_prediction_set[:, start_index: INPUT_LENGTH, :]
    next_INPUT_LENGTH_points = np.concatenate((next_INPUT_LENGTH_minus_ONE_points_slice, chained_predictions), axis=1)
    new_prediction = model.predict(next_INPUT_LENGTH_points)
    new_prediction = new_prediction.reshape(new_prediction.shape + (1,))
    chained_predictions = np.concatenate((chained_predictions, new_prediction), axis=1)
    
  return chained_predictions[-1::]