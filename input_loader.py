import os
import matplotlib.pyplot as plt
import numpy as np


class ImageLoader:

    def __init__(self, path='images/', extension='.jpg'):
        """Loads Images from files"""
        self.path = path
        self.ext = extension
        files = os.listdir(self.path)

        images_plt = [plt.imread(self.path+f) for f in files if f.endswith(self.ext)]

        images_np = np.array(images_plt)
        rdy_img =[]
        for img in images_np:
                scale_img = (1 - np.dot(img, [0.299, 0.587, 0.114])/255)*255
                for row in scale_img:
                    i = 0
                    while i < 28:
                        if row[i] < 5:
                            row[i] = 0
                        i += 1
                rdy_img.append(scale_img)
        self.images = np.array(rdy_img)
        # self.images = (1 - np.dot(images_np, [0.299, 0.587, 0.114])/255)*255

    def getimages(self):
        return self.images