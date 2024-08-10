#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import platform, pydicom, pylab
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from skimage.io import imsave, imread


# In[2]:


def check_dicom(path):
    # Check dicom file
    try:
        dcm_info = pydicom.dcmread(path,force = True)
    except:
        print('dicom error')
        dcm_info = None
    return dcm_info

def check_imagematch(dcm_info):
    # check file in directory
    try:
        return dcm_info.pixel_array
    except:
        print('dicom image error')
        return None

def check_tags(dcmpath):
    files = os.listdir(dcmpath)
    files.sort()
    uid = []
    paths = []
    dcms = []
    for filepath in range(len(files)):
        dcm_path = os.path.join(dcmpath, files[filepath])
        dcm_info = check_dicom(dcm_path)
        dcms.append(dcm_info.SeriesInstanceUID)
#         paths.append(dcm_path)
        if dcm_info is None:
            continue
        try:
            dcm_info.ImageType
        except:
            continue
        if not hasattr(dcm_info,'SliceThickness'):
            all_paths = None
            return all_paths
        if 'DERIVED' in dcm_info.ImageType:
            continue
        if 'AXIAL' not in dcm_info.ImageType:
            continue
        image = check_imagematch(dcm_info)
        if image is None:
            continue
        if image.shape[0] != 512:
            continue
        if len(image.shape) != 2:
            continue
        if dcm_info is None:
            continue
        elif filepath < 2:
            continue
        paths.append(dcm_path)
        uid.append(dcm_info.SeriesInstanceUID)
    if len(uid) == 0:
        all_paths = None
        return all_paths
    if len(set(uid)) != 1 and len(uid) != 0:
        idx = [i for i,  s in enumerate(dcms) if uid[0] in s]
        path_subset = [paths[x] for x in idx]
        all_paths = path_subset.copy()
        return all_paths
    if len(set(uid)) == 1:
        all_paths = paths.copy()
        return all_paths


# In[3]:


def read_dicom(paths):
    # read dicom file and header
    #print(paths)
    for filepath in range(len(paths)):
        ds = pydicom.dcmread(paths[filepath])
        image = ds.pixel_array
        if filepath == 0:
            raw = np.zeros((image.shape[0], image.shape[1],len(paths)-2), dtype = np.float32)
            raw[:,:,0] = image.copy()
        else:
            try:
                raw[:,:,filepath] = image.copy()
            except:
                continue
    raw_img = raw.copy()
    return raw_img, ds


def read_raw_png(paths): # Specs: paths must contain list of full directories NOT folder names
    if len(paths) == 1:
        img = imread(paths[0], as_gray=True)
        return np.expand_dims(img, axis=2)
    img = imread(paths[0], as_gray=True)
    return np.append(np.expand_dims(img, axis=2), read_raw_png(paths[1:]), axis=2)




# In[4]:


def zeroPadding2d(im, desired_size):
    delta_w = desired_size - im.shape[1]
    delta_h = desired_size - im.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    try:
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_im
    except:
        pass

def zeroPadding3d(vol, desired_size):
    delta_w = (desired_size - vol.shape[1])//2
    delta_h = (desired_size - vol.shape[0])//2
    delta_s = (desired_size - vol.shape[2])//2

    new_vol = np.zeros((desired_size, desired_size, desired_size),dtype=vol.dtype)
    print('zero padding: ', np.shape(vol), '->', np.shape(new_vol))
    new_vol[delta_h:delta_h+vol.shape[0],delta_w:delta_w+vol.shape[1],delta_s:delta_s+vol.shape[2]] = vol
    return new_vol



# In[5]:


def resize3d(vol, steps):
    x, y, z = [steps[k] * np.arange(vol.shape[k]) for k in range(3)]  # original grid
    print(np.shape((x,y,z)))
    f = RegularGridInterpolator((x, y, z), vol)    # interpolator

    dx, dy, dz = 1.0, 1.0, 1.0    # new step sizes
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]   # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    new_values = f(new_grid)

    return new_values


# In[6]:


def montageDisp(X,  res_save = 0, folder = None, pn_name = None, image_name = None, colormap=pylab.cm.gist_gray):
    sz = np.shape(X)
    mm = int(np.ceil(np.sqrt(sz[2])))
    nn = mm
    M = np.zeros((mm * sz[0], nn * sz[1]), dtype = X.dtype)

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= sz[2]:
                break
            sliceM, sliceN = j * sz[0], k * sz[1]
            M[sliceN:sliceN + sz[0], sliceM:sliceM + sz[1]] = X[:, :, image_id]
            image_id += 1
    # plt.imshow(np.uint8(M), cmap=colormap)
    # plt.axis('off')
    # plt.show()
    imS = cv2.resize(np.uint8(M), (np.shape(M)[0]//4, np.shape(M)[1]//4))
    if image_name == None :
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 900, 900)
        cv2.imshow('image', imS)
    else:
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_name, 900, 900)
        cv2.imshow(image_name, imS)
    cv2.waitKeyEx(5)

    if res_save == 1:
        if os.path.exists(folder) == False :
            os.mkdir(folder)

        file_name = folder + pn_name +".png"
#         cv2.imwrite(file_name, M)
        imsave(file_name, M)


# In[7]:


def getEig(img):
    y, x = np.nonzero(img)
    # Subtract mean from each dimension
#     print( np.mean(x))
#     print( np.mean(y))
    xx = x - np.mean(x)
    yy = y - np.mean(y)
    coords = np.vstack([xx, yy])
    #  Covariance matrix and its eigenvectors and eigenvalues
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    # Sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
#     print('p1:'+str(x_v1)+' '+str(y_v1))
#     print('p2:'+str(x_v2)+' '+str(y_v2))
    scale = 1
    x_v1 = x_v1 +np.mean(x)
    y_v1 = y_v1 +np.mean(y)
    x_v2 = x_v2 +np.mean(x)
    y_v2 = y_v2 +np.mean(y)
    return np.array([x_v1, y_v1]), np.array([x_v2, y_v2])


# In[8]:


def distLine2Pnt(p1, p2, p3):
    d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    return d

def saveResult(data_result, dir_name, file_name):
    data_result = np.multiply(data_result, 1/255)
    data_result = data_result.astype(np.uint8)
    data_result = zeroPadding3d(data_result, 512)
    print(data_result.shape, data_result.dtype)
    filename = dir_name+file_name+'.raw'
    data_result.tofile(filename)

def save_result_as_png(mask_volume, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, file_name)):
        os.mkdir(os.path.join(save_dir, file_name))

    volume_size = np.shape(mask_volume)

    for indexer in range(volume_size[2]):
        imsave(
            os.path.join(save_dir,
                         file_name,
                         (file_name + '_' + str(indexer) + '.png')),
            mask_volume[:, :, indexer].astype(np.uint8),
            check_contrast=False
        )