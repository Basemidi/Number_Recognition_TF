import ML_Number_Reco as HW
import model_loader as mload
import input_loader as inp
import matplotlib.pyplot as plt
import numpy as np

class Main():
    # Main halt
    def __init__(self):
        """
        :rtype: none
        """
        # HW.BuildNumberReco()
        x_test, y_test = mload.MnistLoader().test_set()

        network = mload.Loader().getmodel()
        new_test = inp.ImageLoader().getimages()
        #  Test sequenz
        # print(network.predict(new_test[4:]))
        print(x_test[5])
        plt.imshow(new_test[2], cmap='Greys')
        plt.show()



Main()
