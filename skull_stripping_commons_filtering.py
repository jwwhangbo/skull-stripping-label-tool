#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import ndimage

from itertools import combinations


# In[2]:


def get3dMasks():
    h = []
    for num_h in range(26):
        temp = np.zeros((3,3,3), dtype=int)
        temp[1][1][1] = -1
        h.append(temp)
        
    h[0][0][2][2] = 1
    h[1][1][2][2] = 1
    h[2][2][2][2] = 1
    h[3][0][1][2] = 1
    h[4][2][1][2] = 1
    h[5][0][0][2] = 1
    h[6][1][0][2] = 1
    h[7][2][0][2] = 1
    
    h[8][0][2][1] = 1
    h[9][2][2][1] = 1
    h[10][0][0][1] = 1
    h[11][2][0][1] = 1
    
    h[12][0][2][0] = 1
    h[13][1][2][0] = 1
    h[14][2][2][0] = 1
    h[15][0][1][0] = 1
    h[16][2][1][0] = 1
    h[17][0][0][0] = 1
    h[18][1][0][0] = 1
    h[19][2][0][0] = 1
    
    h[20][0][1][1] = 1
    h[21][2][1][1] = 1
    h[22][1][2][1] = 1
    h[23][1][0][1] = 1
    h[24][1][1][2] = 1
    h[25][1][1][0] = 1
    
    return h


# In[3]:


def getDiffusionFnc1(nabla, kappa):
    c = []
    sz = np.shape(nabla)
    for num_c in range(sz[0]):
        temp = np.exp(-(nabla[num_c]/kappa)*(nabla[num_c]/kappa))
        c.append(temp)
    return c

def getDiffusionFnc2(nabla, kappa):
    c = []
    sz = np.shape(nabla)
    for num_c in range(sz[0]):
        temp = 1/(1+((nabla[num_c]/kappa)*(nabla[num_c]/kappa)))
        c.append(temp)
    return c


# In[4]:


def anisodiff3d(vol, info, num_itr=2, delta_t=3/44, kappa=50, option=1):
    # (frame, row, column)
    sz = np.shape(vol)
    # initial condition = vol (dtype = double)
    diff_vol = np.float64(vol)
    
    # center voxel distances
    x = info.PixelSpacing[0]
    y = info.PixelSpacing[1]
    z = info.SliceThickness
    dx, dy = 1, 1
    dz = z/x
    dd = np.sqrt(dx*dx+dy*dy)
    dh = np.sqrt(dx*dx+dz*dz)
    dc = np.sqrt(dd*dd+dz*dz)
    
    # 3d convolution masks - finite differences
    h = get3dMasks()
    nabla = []
    for t in range(num_itr):
        # Finite differences. [imfilter(.,.,'conv') can be replaced by convn(.,.,'same')]
        # Due to possible memory limitations, the diffusion will be calculated at each page/slice of the volume.
        for p in range(sz[2]-2):
            diff3pp = diff_vol[:,:,p:p+2]
            for num_h in range(26):
                aux = ndimage.convolve(diff3pp, h[num_h], mode='nearest')
                nabla.append(aux[:,:,1])
            if option == 1:
                c = getDiffusionFnc1(nabla, kappa)
            elif option == 2:
                c = getDiffusionFnc2(nabla, kappa)
            
            diff_vol[:,:,p+1] = (diff_vol[:,:,p+1] + delta_t *                                     (1/(dz*dz))*c[25]*nabla[25] + (1/(dz*dz))*c[24]*nabla[24] +                                     (1/(dx*dx))*c[23]*nabla[23] + (1/(dx*dx))*c[22]*nabla[22] +                                     (1/(dy*dy))*c[21]*nabla[21] + (1/(dz*dz))*c[20]*nabla[20] +             
                                    (1/(dc*dc))*c[19]*nabla[19] + (1/(dh*dh))*c[18]*nabla[18] + \
                                    (1/(dc*dc))*c[17]*nabla[17] + (1/(dh*dh))*c[16]*nabla[16] + \
                                    (1/(dh*dh))*c[15]*nabla[15] + (1/(dc*dc))*c[14]*nabla[14] + \
                                    (1/(dh*dh))*c[13]*nabla[13] + (1/(dc*dc))*c[12]*nabla[12] + \
            
                                    (1/(dd*dd))*c[11]*nabla[11] + (1/(dd*dd))*c[10]*nabla[10] + \
                                    (1/(dd*dd))*c[9]*nabla[9] + (1/(dd*dd))*c[8]*nabla[8] + \
            
                                    (1/(dc*dc))*c[7]*nabla[7] + (1/(dh*dh))*c[6]*nabla[6] + \
                                    (1/(dc*dc))*c[5]*nabla[5] + (1/(dh*dh))*c[4]*nabla[4] + \
                                    (1/(dh*dh))*c[3]*nabla[3] + (1/(dc*dc))*c[2]*nabla[2] + \
                                    (1/(dh*dh))*c[1]*nabla[1] + (1/(dc*dc))*c[0]*nabla[0] )
    
    return diff_vol


# In[5]:


def unsharp_filter(img, sigma=3, ratio=0.7):
    # Gaussian filtering 
    img_mf = ndimage.filters.gaussian_filter(img, sigma)
    # Calculate the Laplacian
    img_la = ndimage.filters.laplace(img_mf)
    # sharpened image
    result = img - ratio * img_la
    return result
