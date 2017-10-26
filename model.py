import numpy as np
import cv2

# disable tensorflow debug information
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Cropping2D, Dense, Lambda, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

from keras import backend as K
from sklearn.model_selection import train_test_split


def marcin(l2_val=0, use_batch_normalization=False):
    model = Sequential()

    # normalize the input
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(66, 200, 3)))

    # convolutional layers
    model.add(Conv2D(24, (5, 5), strides=2, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), strides=2, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), strides=2, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
        model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
       model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
       model.add(BatchNormalization())

    # fully connected layers
    model.add(Flatten())

    model.add(Dense(100, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
        model.add(BatchNormalization())

    model.add(Dense(50, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
       model.add(BatchNormalization())

    model.add(Dense(10, activation='elu', kernel_regularizer=l2(l2_val)))
    if use_batch_normalization:
       model.add(BatchNormalization())

    model.add(Dense(1, activation='tanh'))

    return model



if __name__ == "__main__":
    X_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    # split data into training/validation
    X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.4, random_state=43)
    split = int(len(X_test)/2)
    X_valid = X_test[:split]
    y_valid = y_test[:split]
    X_test = X_test[split:]
    y_test = y_test[split:]

    # use server to increase performance (4x GPUs)
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)

    model = marcin(l2_val=0, use_batch_normalization=True)
    model.compile(loss='mse', optimizer='adam')

    # add early stopping and checkpoints saving
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    callbacks_list = [checkpoint, stopping]

    K.set_session(sess)
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), shuffle=True, callbacks=callbacks_list)

    # evaluate
    eval_loss = model.evaluate(X_test, y_test, verbose=1)
    print(eval_loss)
