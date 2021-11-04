"""
Defination of NN GRU model
"""
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import GRU
from keras.models import Sequential

def build_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model