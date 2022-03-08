from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

def load_imgs():
    imgset = []
    for filename in os.listdir(r"./images_small"):
        img = np.array(Image.open("./images_small/"+filename))
        img = (img / float(img.max())).astype(np.float32)
        imgset.append(img)
    return imgset

if __name__=='__main__':
    imgset = load_imgs()
    plt.imshow(imgset[50])
    plt.title('image of shape {}'.format(imgset[50].shape))
    plt.show()
