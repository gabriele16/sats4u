import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow as tf
import datetime
import pickle
from sats4u.candles2timeseries import denorm


class TimeSeries2Model:
    def __init__(self, x_candles, x_time, y, scaler,
                 target="Close", split_fraction=0.9, epochs=20, batch_size=4096):

        what_to_predict = ['Close', 'LogReturns', 'UpDown']

        if target not in what_to_predict:
            raise ValueError(
                "Invalid target to predict, Expected one of: %s" % what_to_predict)

        self.target = target
        self.x_candles = x_candles
        self.x_time = x_time
        self.y = y
        self.split_fraction = split_fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = scaler

        # self.x_train_candles.shape[1], self.x_train_candles.shape[2]

        self.params = {'features_dimensions': [np.shape(self.x_candles[0])[0], np.shape(self.x_candles[0])[1]],
                       'ae_num_labels': 5,
                       'ae_hidden_units': [96, 96, 896, 448, 448, 256],
                       'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436,
                                         0.10431484318345882, 0.49230389137187497, 0.32024444956111164,
                                         0.2716856145683449, 0.4379233941604448],
                       'ls': 0,
                       'lr': 1e-3,
                       'kernel_sizes': [3, 7, 13],
                       'filter_size_1': 32,
                       'filter_size_2': 64,
                       'lstm_units': 8,
                       'lstm_dense_units':  128,
                       }

    def train_test_split(self, train_whole=False):

        split_point = int(len(self.x_candles) * self.split_fraction)
        self.split_point = split_point

        if self.split_fraction == 0.0 or self.split_fraction == 1.0 or train_whole == True:

            self.split_point = 0
            self.x_train_candles = np.asarray(self.x_candles, dtype=np.float32)
            self.x_train_time = np.asarray(self.x_time, dtype=np.float32)
            self.y_train = np.asarray(self.y, dtype=np.float32)
            self.x_test_candles = self.x_train_candles.copy()
            self.x_test_time = self.x_train_time.copy()
            self.y_test = self.y_train.copy()

        else:

            self.x_train_candles = np.asarray(
                self.x_candles[:split_point], dtype=np.float32)
            self.x_train_time = np.asarray(
                self.x_time[:split_point], dtype=np.float32)
            self.y_train = np.asarray(self.y[:split_point], dtype=np.float32)
            self.x_test_candles = np.asarray(
                self.x_candles[split_point:], dtype=np.float32)
            self.x_test_time = np.asarray(
                self.x_time[split_point:], dtype=np.float32)
            self.y_test = np.asarray(self.y[split_point:], dtype=np.float32)

    def initialize_model_params(self, features_dimensions, ae_hidden_units, dropout_rates,
                                ae_num_labels=5, ls=1e-2, lr=1e-3,
                                kernel_sizes=[3, 7, 13], filter_size_1=32,
                                filter_size_2=64, lstm_units=8, lstm_dense_units=128
                                ):

        self.features_dimensions = features_dimensions
        self.ae_hidden_units = ae_hidden_units
        self.dropout_rates = dropout_rates
        self.ae_num_labels = ae_num_labels
        self.ls = ls
        self.lr = lr

        self.kernel_sizes = kernel_sizes
        self.filter_size_1 = filter_size_1
        self.filter_size_2 = filter_size_2
        self.lstm_units = lstm_units
        self.lstm_dense_units = lstm_dense_units

        self.k0 = self.kernel_sizes[0]
        self.k1 = self.kernel_sizes[1]
        self.k2 = self.kernel_sizes[2]

    def create_ae(self):

        #        inp = tf.keras.layers.Input(shape=(num_columns, ))
        input_candles_ae = keras.Input(
            shape=(
                self.features_dimensions[0], self.features_dimensions[1]), name="candles_ae"
        )
        x0 = tf.keras.layers.BatchNormalization()(input_candles_ae)

        encoder = tf.keras.layers.GaussianNoise(self.dropout_rates[0])(x0)
        encoder = tf.keras.layers.Dense(self.ae_hidden_units[0])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('swish')(encoder)

        decoder = tf.keras.layers.Dropout(self.dropout_rates[1])(encoder)
    #    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)
        decoder = tf.keras.layers.Dense(shape=(
            self.features_dimensions[0], self.features_dimensions[1]), name='decoder')(decoder)

        x_ae = tf.keras.layers.Dense(self.ae_hidden_units[1])(decoder)
        x_ae = tf.keras.layers.BatchNormalization()(x_ae)
        x_ae = tf.keras.layers.Activation('swish')(x_ae)
        x_ae = tf.keras.layers.Dropout(self.dropout_rates[2])(x_ae)

        out_ae = tf.keras.layers.Dense(
            self.num_labels, activation='sigmoid', name='ae_action')(x_ae)

        return out_ae, x0, encoder, decoder

    def create_ae_mlp_model(self, out_ae, x0, encoder, decoder):

        out_ae, x0, encoder, decoder = self.create_ae()

        x = tf.keras.layers.Concatenate()([x0, encoder])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rates[3])(x)

        for i in range(2, len(self.ae_hidden_units)):
            x = tf.keras.layers.Dense(self.ae_hidden_units[i])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('swish')(x)
            x = tf.keras.layers.Dropout(self.dropout_rates[i + 2])(x)

        out = tf.keras.layers.Dense(
            self.num_labels, activation='sigmoid', name='action')(x)

        model = tf.keras.models.Model(
            inputs=self.input_candles_ae, outputs=[decoder, out_ae, out])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss={'decoder': tf.keras.losses.MeanSquaredError(),
                            'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing=self.ls),
                            'action': tf.keras.losses.BinaryCrossentropy(label_smoothing=self.ls),
                            },
                      metrics={'decoder': tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                               'ae_action': tf.keras.metrics.AUC(name='AUC'),
                               'action': tf.keras.metrics.AUC(name='AUC'),
                               },
                      )
        return model

    def create_conv_lstm_block(self, input, kernel_size_1, kernel_size_2):

        conv_1 = keras.layers.Conv1D(
            filters=self.filter_size_1, kernel_size=kernel_size_1, activation=keras.activations.swish, padding="same"
        )(input)
        average_1 = keras.layers.AveragePooling1D()(conv_1)
        conv_2 = keras.layers.Conv1D(
            filters=self.filter_size_2, kernel_size=kernel_size_2, activation=keras.activations.swish, padding="same"
        )(average_1)
        average_2 = keras.layers.AveragePooling1D()(conv_2)
        lstm_1 = keras.layers.LSTM(
            units=self.filter_size_2, return_sequences=True)(average_2)
        lstm_2 = keras.layers.LSTM(units=self.filter_size_2)(lstm_1)

        return lstm_2

    def create_lstm_cnn_model(self):

        input_candles = keras.Input(
            shape=(
                self.x_train_candles.shape[1], self.x_train_candles.shape[2]), name="candles"
        )
        input_time = keras.Input(
            shape=(self.x_train_time.shape[1], self.x_train_time.shape[2]), name="time")

        conv_1 = self.create_conv_lstm_block(
            input_candles, kernel_size_1=self.k0, kernel_size_2=self.k0)
        conv_2 = self.create_conv_lstm_block(
            input_candles, kernel_size_1=self.k1, kernel_size_2=self.k1)
        conv_3 = self.create_conv_lstm_block(
            input_candles, kernel_size_1=self.k2, kernel_size_2=self.k2)

        lstm_time_1 = keras.layers.LSTM(
            units=self.lstm_units, return_sequences=True)(input_time)
        lstm_time_2 = keras.layers.LSTM(units=self.lstm_units)(lstm_time_1)

        conc = keras.layers.Concatenate(
            axis=-1)([conv_1, conv_2, conv_3, lstm_time_2])

        if self.target == "UpDown":
            dense_1 = keras.layers.Dense(
                units=self.lstm_dense_units, activation=keras.activations.swish)(conc)
            dense_2 = keras.layers.Dense(
                units=self.lstm_dense_units, activation=keras.activations.swish)(dense_1)            
            output = keras.layers.Dense(
                units=1, activation=keras.activations.sigmoid)(dense_2)
            self.model = keras.Model(
                inputs=[input_candles, input_time], outputs=output)
            self.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=tf.keras.losses.BinaryCrossentropy(),
                               metrics=['accuracy', tf.keras.metrics.AUC()])
        else:
            dense_1 = keras.layers.Dense(
                units=self.lstm_dense_units, activation=keras.activations.swish)(conc)
            dense_2 = keras.layers.Dense(
                units=self.lstm_dense_units, activation=keras.activations.swish)(dense_1)
            output = keras.layers.Dense(
                units=1, activation=keras.activations.linear)(dense_2)
            self.model = keras.Model(
                inputs=[input_candles, input_time], outputs=output)
            self.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=keras.losses.mean_absolute_error)

    def sats2model(self):

        self.train_test_split()
        self.initialize_model_params(**self.params)
        self.create_lstm_cnn_model()
        keras.utils.plot_model(
            self.model, "conv_lstm_net.png", show_shapes=True)

    def sats2train(self, model_name, save_model=True, epochs=20):

        self.epochs = epochs
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="model/weights", save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.history = self.model.fit(
            [self.x_train_candles, self.x_train_time],
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(
                [self.x_test_candles, self.x_test_time], self.y_test),
            callbacks=model_checkpoint_callback
        )

        self.model.load_weights("model/weights")

        if save_model:
            self.model.save(model_name)
            scalerfile = model_name + "/scaler.sav"
            pickle.dump(self.scaler, open(scalerfile, "wb"))

    def load_model(self, model_name):

        self.model = keras.models.load_model(model_name)

    def load_scaler(self, scaler_name):

        self.scaler = pickle.load(open(scaler_name, "rb"))

    def sats2pred(self, predict_on_test=True):

        if predict_on_test:
            self.preds = self.model.predict(
                [self.x_test_candles, self.x_test_time], batch_size=self.batch_size)
        else:
            self.x_candles = np.asarray(self.x_candles, dtype=np.float32)
            self.x_time = np.asarray(self.x_time, dtype=np.float32)
            self.preds = self.model.predict(
                [self.x_candles, self.x_time], batch_size=self.batch_size)
        if self.target == "UpDown":
            self.preds = (self.preds > 0.5).astype(int)
