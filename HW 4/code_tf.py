from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


########################## Helper functions and classes ###############################
def load_data(path):
    data = np.load(path)
    x = data['x']
    y = data['y']
    locations = data['locations']
    times = data['times']

    data.close()

    return x, y, locations, times


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def export_result(fname, rmse):
    with open(fname, mode='w') as f:
        f.write("RMSE is: {:.10f}".format(rmse))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


############################ HW 4 starts here ######################################
def q1():
    x, y, locations, times = load_data('./norm_data/val.npz')

    mid = np.ravel_multi_index((3, 3), (7, 7))
    # Historical average
    hist_seq = x[:, :, mid]
    y_pred = np.mean(hist_seq, axis=-1)

    print('\nUsing Historical Mean:')
    results = rmse(y, y_pred)
    print('    Root Mean Square Error is: {:.6f}'.format(results))
    export_result('./norm_results/q1.txt', results)


def q2():
    x, y, locations, times = load_data('./norm_data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./norm_data/val.npz')

    X_train = x.reshape(len(x), -1)
    X_val = x_val.reshape(len(x_val), -1)
    reg = LinearRegression(n_jobs=-1)
    reg.fit(X_train, y)
    pred = reg.predict(X_val)

    print('\nUsing Linear Regression:')
    print('    Root Mean Square Error is: {:.6f}'.format(rmse(y_val, pred)))
    export_result('./norm_results/q2.txt', rmse(y_val, pred))


def q3():
    x, y, locations, times = load_data('./norm_data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./norm_data/val.npz')

    X_train = x.reshape(len(x), -1)
    X_val = x_val.reshape(len(x_val), -1)
    reg = XGBRegressor(learning_rate=0.1)
    reg.fit(X_train, y)
    pred = reg.predict(X_val)

    print('\nUsing XGBoost:')
    print('    Root Mean Square Error is: {:.6f}'.format(rmse(y_val, pred)))
    export_result('./norm_results/q3.txt', rmse(y_val, pred))


def q4():
    save_path = './norm_save/q4/'

    x, y, locations, times = load_data('./norm_data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./norm_data/val.npz')

    model = tf.keras.Sequential([
        layers.SimpleRNN(128, input_shape=(x.shape[1], x.shape[2])),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)]

    # model.fit(x, y, batch_size=32, epochs=100, callbacks=callbacks, validation_data=(x_val, y_val))

    # Test
    model.load_weights(save_path)
    y_hat = model.predict(x_val)
    result = rmse(y_val, y_hat)

    export_result('./norm_results/q4.txt', result)


def q5():
    save_path = './norm_save/q5/'

    x, y, locations, times = load_data('./norm_data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./norm_data/val.npz')

    model = tf.keras.Sequential([
        layers.LSTM(128, input_shape=(x.shape[1], x.shape[2])),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)]

    model.fit(x, y, batch_size=32, epochs=100, callbacks=callbacks, validation_data=(x_val, y_val))

    # Test
    model.load_weights(save_path)
    y_hat = model.predict(x_val)
    result = rmse(y_val, y_hat)

    export_result('./norm_results/q5.txt', result)


class Q6Model(tf.keras.Model):
    def __init__(self, step, edge, dropout=0.0):
        super(Q6Model, self).__init__()
        self.convlstm = tf.keras.Sequential([
            layers.ConvLSTM2D(256, 3, input_shape=(step, edge, edge, 1), return_sequences=True, dropout=dropout),
            layers.ConvLSTM2D(128, 3, dropout=dropout),
            layers.Dropout(dropout),
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu')
        ])

        self.fcs = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        x, t, loc = inputs
        out = self.convlstm(x)
        out = tf.concat([out, t, loc], axis=1)
        out = self.fcs(out)

        return out




def q6():
    save_path = './norm_save/q6/'

    x, y, locations, times = load_data('./norm_data/train.npz')
    x = x.reshape(-1, x.shape[1], 7, 7)
    locations = locations.reshape(x.shape[0], -1)
    x_val, y_val, loc_val, t_val = load_data('./norm_data/val.npz')
    x_val = x_val.reshape(-1, x.shape[1], 7, 7)
    loc_val = loc_val.reshape(x_val.shape[0], -1)

    # Nets
    dropout = 0.25
    x_in = tf.keras.Input(shape=(x.shape[1], 7, 7, 1))
    t_in = tf.keras.Input(shape=(1,))
    loc_in = tf.keras.Input(shape=(2,))

    out = layers.ConvLSTM2D(256, 3, return_sequences=True)(x_in)
    out = layers.Dropout(dropout)(out)
    out = layers.ConvLSTM2D(128, 3, dropout=dropout)(out)
    out = layers.Dropout(dropout)(out)
    out = layers.Flatten()(out)
    out = layers.Dense(256, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)

    out = layers.Concatenate(axis=-1)([out, t_in, loc_in])
    out = layers.Dense(256, activation='relu')(out)
    out = layers.Dense(1)(out)

    # model = Q6Model(step=x.shape[1], edge=7, dropout=0.25)
    model = tf.keras.Model(inputs=[x_in, t_in, loc_in], outputs=out)

    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)]

    model.fit([x, times, locations], y, batch_size=32, epochs=100, callbacks=callbacks,
              validation_data=([x_val, t_val, loc_val], y_val))

    # Test
    model.load_weights(save_path)
    y_hat = model.predict([x_val, t_val, loc_val])
    result = rmse(y_val, y_hat)

    export_result('./norm_results/q6.txt', result)


if __name__ == '__main__':
    # q1()
    # q2()
    # q3()
    # q4()
    # q5()
    q6()