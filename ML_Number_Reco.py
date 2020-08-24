import tensorflow as tf
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt


class BuildNumberReco():
    # Just builds the network Trains it and saves the model for later use
    def __init__(self) -> object:
        """
        :rtype: object
        """

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, y_train = x_train/255, y_train

        network = tf.keras.models.Sequential(
                [tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                              ])

        tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

        network.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        network.fit(x_train, y_train, epochs= 5 , callbacks=[tensorboard])

        print('Evaluations starts')
        test_loss, test_acc = network.evaluate(x_test, y_test)
        print('Test accuarcy : ', test_acc)
        print('Test Loss :', test_loss)
        network.save('Model/net.h5')
