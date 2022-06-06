import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf

def swish(x):
    return keras.backend.sigmoid(x) * x

class TrainTimeSeries():

    def __init__(self, x_candles, x_time, y, split_fraction = 0.9):

        self.x_candles = x_candles
        self.x_time = x_time
        self.y = y
        self.split_fraction = split_fraction
    
    def train_test_split(self):

        split_point = int(len(self.x_candles) * self.split_fraction)       
        self.x_train_candles = np.asarray(self.x_candles[:split_point], dtype=np.float32)
        self.x_train_time = np.asarray(self.x_time[:split_point], dtype=np.float32)
        self.y_train = np.asarray(self.y[:split_point], dtype=np.float32)

        self.x_test_candles = np.asarray(self.x_candles[split_point:], dtype=np.float32)
        self.x_test_time = np.asarray(self.x_time[split_point:], dtype=np.float32)
        self.y_test = np.asarray(self.y[split_point:], dtype=np.float32)


    def get_conv_lstm_block(self,input,kernel_size_1,kernel_size_2):

        conv_1 = keras.layers.Conv1D(
            filters=self.filter_size_1,
            kernel_size=kernel_size_1,
            activation=keras.activations.swish,
            padding='same'
        )(input)
        average_1 = keras.layers.AveragePooling1D()(conv_1)
        
        conv_2 = keras.layers.Conv1D(
            filters=self.filter_size_2,
            kernel_size=kernel_size_2,
            activation=keras.activations.swish,
            padding='same'
        )(average_1)
        average_2 = keras.layers.AveragePooling1D()(conv_2)
        
        lstm_1 = keras.layers.LSTM(units=self.filter_size_2, return_sequences=True)(average_2)
        lstm_2 = keras.layers.LSTM(units=self.filter_size_2)(lstm_1)
        
        return lstm_2

    def lstm_cnn_model(self,kernel_sizes = [3,7,13], filter_size_1 = 32, filter_size_2 = 64, lstm_units = 8, dense_units = 128):

        self.kernel_sizes = kernel_sizes
        self.filter_size_1 = filter_size_1
        self.filter_size_2 = filter_size_2
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        k0 = self.kernel_sizes[0]
        k1 = self.kernel_sizes[1]
        k2 = self.kernel_sizes[2]

        input_candles = keras.Input(shape=(self.x_train_candles.shape[1], self.x_train_candles.shape[2]), name='Candles')
        input_time = keras.Input(shape=(self.x_train_time.shape[1], self.x_train_time.shape[2]), name='Time')

        conv_1 = self.get_conv_lstm_block(input_candles,kernel_size_1=k0,kernel_size_2=k0)
        conv_2 = self.get_conv_lstm_block(input_candles,kernel_size_1=k1,kernel_size_2=k1)
        conv_3 = self.get_conv_lstm_block(input_candles,kernel_size_1=k2,kernel_size_2=k2)

        lstm_time_1 = keras.layers.LSTM(units=self.lstm_units, return_sequences=True)(input_time)
        lstm_time_2 = keras.layers.LSTM(units=self.lstm_units)(lstm_time_1)

        conc = keras.layers.Concatenate(axis=-1)([conv_1, conv_2, conv_3, lstm_time_2])

        dense_1 = keras.layers.Dense(units=self.dense_units, activation=keras.activations.swish)(conc)
        dense_2 = keras.layers.Dense(units=self.dense_units, activation=keras.activations.swish)(dense_1)

        output = keras.layers.Dense(units=1, activation=keras.activations.linear)(dense_2)

        self.model = keras.Model(inputs=[input_candles, input_time], outputs=output)

        self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_absolute_error)

        self.model.summary()
        keras.utils.plot_model(self.model, "conv_lstm_net.png", show_shapes=True)

    
    def sats2model(self):

        self.train_test_split()
        self.lstm_cnn_model()