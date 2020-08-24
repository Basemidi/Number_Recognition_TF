import tensorflow as tf


class Loader():

    def __init__(self, path='Model/net.h5'):
        """
        :rtype:model
        """
        self.path = path
        self.model = tf.keras.models.load_model(filepath=self.path)


    def getmodel(self):
        return self.model

class MnistLoader():

    def __init__(self):
        """
        :rtype:Dataset
        """
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_test = x_test
        self.y_test = y_test
        self.x_train, self.y_train = x_train/255, y_train

    def test_set(self):
        return self.x_test, self.y_test

    def train_set(self):
        return self.x_train, self.y_train