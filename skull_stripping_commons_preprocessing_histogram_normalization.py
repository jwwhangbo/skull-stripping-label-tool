from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure
from skimage.morphology import disk
from skimage import filters
from commons.preprocessing_image_entropy import preprocessing2D as preproc2D

def preprocessing2D(img):
    p2, p98 = np.percentile(img, (2, 98))
    new_img = exposure.rescale_intensity(img, in_range=(p2, p98))
    new_img = exposure.adjust_gamma(new_img, gamma=2)
    processed = preproc2D(new_img)

    return processed

def preprocessing3D(vol):
    new_vol = np.zeros(vol.shape)
    inv_vol = np.zeros(vol.shape)
    for frame in range(vol.shape[2]):
        #         print('frame: '+str(frame))
        new_vol[:, :, frame] = preprocessing2D(vol[:, :, frame])
    return new_vol
