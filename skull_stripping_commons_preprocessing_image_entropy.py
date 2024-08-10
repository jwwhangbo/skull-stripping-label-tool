from __future__ import print_function

import numpy as np
from skimage import filters, morphology, exposure
import matplotlib.pyplot as plt


# In[2]:


def preprocessing2D(img, disk_sz=3):
    img_equalized = exposure.equalize_hist(img)
    img_sigmoid_adjusted = exposure.adjust_sigmoid(img_equalized,
                                                   cutoff=((img_equalized.max() - img_equalized.min()) / img_equalized.max()))
    # Get image entropy (input image should be in a range [-1 1])
    max_val = img_sigmoid_adjusted.max()
    new_img = filters.rank.entropy(img_sigmoid_adjusted / max_val, morphology.disk(disk_sz))

    # normalization
    min_entropy = new_img.min()
    max_entropy = new_img.max()
    #     print(min_entropy, max_entropy)
    new_max = 255
    new_img = (new_img - min_entropy) * new_max / (max_entropy - min_entropy)
    new_img[new_img > new_max] = new_max
    new_img[new_img < 0] = 0
    inv_new_img = new_max - new_img

    # data type conversion from float64 to uint32
    inv_new_img.astype(np.float32)

    #     plt.imshow(new_img)
    #     plt.title('filtered image')
    #     plt.show()

    #     print(inv_new_img)
    #     plt.imshow(inv_new_img)
    #     plt.title('inverse image')
    #     plt.show()
    return inv_new_img


# In[3]:


def preprocessing3D(vol):
    new_vol = np.zeros(vol.shape)
    inv_vol = np.zeros(vol.shape)
    for frame in range(vol.shape[2]):
        #         print('frame: '+str(frame))
        new_vol[:, :, frame] = preprocessing2D(vol[:, :, frame])
    return new_vol
