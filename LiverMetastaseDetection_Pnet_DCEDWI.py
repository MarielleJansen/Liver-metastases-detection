# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:25:37 2018

@author: Mariëlle Jansen

Liver lesion segmentation P-net with 2 networks for DCE and DWI

"""
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras import layers, models, utils
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint 
from tensorflow.python.keras._impl.keras import backend as K
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from os import path
import scipy.ndimage.interpolation as sp
from scipy.stats import mode

def read_image(filename):
    nii = nib.load(filename)  # Use nibabel to load nifti image
    data = nii.get_data()
    data = np.asarray(data)
    #data = np.swapaxes(data, 0,1)
    return data   

# Load training data (DCE-MR, DW-MR, and lesion mask)
# Data has been pre-processed by standardization and only selecting slices with 
# lesion present
def load_data_train(basedir, number):
    basedir = basedir
    # Load train data; find body mask and normalize
    t = number
    nameImage='trainDCEMRI'+str(t)+'.nii'
    image = read_image(path.join(basedir, nameImage))
    
    Im1 = image[:,:,:,0]
    Im1 = Im1[:,:,:,np.newaxis]
    Im2 = np.mean(image[:,:,:,1:6], axis=3)
    Im2 = Im2[:,:,:,np.newaxis]
    Im3 = np.mean(image[:,:,:,6:10], axis=3)
    Im3 = Im3[:,:,:,np.newaxis]
    Im4 = np.mean(image[:,:,:,10:12], axis=3)
    Im4 = Im4[:,:,:,np.newaxis]
    Im5 = np.mean(image[:,:,:,12:15], axis=3)
    Im5 = Im5[:,:,:,np.newaxis]
    Im6 = image[:,:,:,15]
    Im6 = Im6[:,:,:,np.newaxis]
    
    Im = np.concatenate((Im1,Im2,Im3,Im4,Im5,Im6), axis=3)
    Im = np.swapaxes(Im, 0,2)
    
    nameImage='trainDWI'+str(t)+'.nii'
    image = read_image(path.join(basedir, nameImage))
    ImDWI = np.swapaxes(image, 0, 2)
    
    mask = read_image(path.join(basedir, 'trainLesion'+str(t)+'.nii'))
    mask = np.swapaxes(mask, 0, 2)
    
    return Im, ImDWI, mask

# Load validation data (DCE-MR, DW-MR, and lesion mask)
# Data has been pre-processed by standardization and only selecting slices with 
# lesion present
def load_data_val(basedir, number):
    basedir = basedir
    t = number
    # Load train data
    nameImage='valDCEMRI'+str(t)+'.nii'
    image = read_image(path.join(basedir, nameImage))
    
    Im1 = image[:,:,:,0]
    Im1 = Im1[:,:,:,np.newaxis]
    Im2 = np.mean(image[:,:,:,1:6], axis=3)
    Im2 = Im2[:,:,:,np.newaxis]
    Im3 = np.mean(image[:,:,:,6:10], axis=3)
    Im3 = Im3[:,:,:,np.newaxis]
    Im4 = np.mean(image[:,:,:,10:12], axis=3)
    Im4 = Im4[:,:,:,np.newaxis]
    Im5 = np.mean(image[:,:,:,12:15], axis=3)
    Im5 = Im5[:,:,:,np.newaxis]
    Im6 = image[:,:,:,15]
    Im6 = Im6[:,:,:,np.newaxis]
    
    Im = np.concatenate((Im1,Im2,Im3,Im4,Im5,Im6), axis=3)
    Im = np.swapaxes(Im, 0,2)
    
    nameImage='valDWI'+str(t)+'.nii'
    image = read_image(path.join(basedir, nameImage))
    ImDWI = np.swapaxes(image, 0, 2)
    
    mask = read_image(path.join(basedir, 'valLesion'+str(t)+'.nii'))
    mask = np.swapaxes(mask, 0,2)
    
    return Im, ImDWI, mask

def standardization(image):
    s1 = 0  # minimum value mapping
    s2 = 1  # maximum value mapping
    pc2 = 0.998 # maximum landmark 99.8th percentile
    
    X = np.ndarray.flatten(image)
    X_sorted = np.sort(X)
    
    p1 = 0 
    p2 = X_sorted[np.round(len(X)*pc2+1).astype(int)]
    st = (s2-s1)/(p2-p1)
    
    image_mapped = np.zeros(X.shape,dtype='float32')
    X.astype('float32')
    image_mapped = s1 + X*st
    
    image_mapped[np.where(image_mapped<0)] = 0
    meanIm = np.mean(image_mapped)
    stdIm = np.std(image_mapped)
    
    Im = (image_mapped-meanIm)/stdIm    # zero mean unit variance for Neural network
    
    Im = np.reshape(Im, image.shape)
    Im.astype('float32')
    return Im

# Load data (DCE-MR, DW-MR, lesion mask, and liver mask)
def load_data(basedir, nameImage, number):
    basedir = basedir
    # Load train data; find body mask and normalize
    mask = read_image(path.join(basedir, str(number),'3DLesionAnnotations.nii'))
    mask = np.swapaxes(mask, 0,2)
    liverMask = read_image(path.join(basedir, str(number),'LiverMask_dilated.nii'))
    liverMask = np.swapaxes(liverMask, 0,2)
    liverMask = liverMask+mask
    idx = liverMask > 0
    liverMask[idx] = 1
    
    image = read_image(path.join(basedir, str(number),nameImage))
    normImage = standardization(image)   
    
    Im1 = normImage[:,:,:,0]
    Im1 = Im1[:,:,:,np.newaxis]
    Im2 = np.mean(normImage[:,:,:,1:6], axis=3)
    Im2 = Im2[:,:,:,np.newaxis]
    Im3 = np.mean(normImage[:,:,:,6:10], axis=3)
    Im3 = Im3[:,:,:,np.newaxis]
    Im4 = np.mean(normImage[:,:,:,10:12], axis=3)
    Im4 = Im4[:,:,:,np.newaxis]
    Im5 = np.mean(normImage[:,:,:,12:15], axis=3)
    Im5 = Im5[:,:,:,np.newaxis]
    Im6 = normImage[:,:,:,15]
    Im6 = Im6[:,:,:,np.newaxis]
    
    Im = np.concatenate((Im1,Im2,Im3,Im4,Im5,Im6), axis=3)
    Im = np.swapaxes(Im, 0,2)

    image = read_image(path.join(basedir, str(number), 'DWI_reg.nii'))
    normImage = standardization(image) 
    ImDWI = np.swapaxes(normImage, 0, 2)

    return Im, ImDWI, mask, liverMask

# define network
def build_network(Inputshape1, Inputshape2, num_class):
    concat_axis = 3
    inputsDCE = layers.Input(shape = Inputshape1)
    inputsDWI = layers.Input(shape = Inputshape2)
    # DCE MRI
    #block 1
    conv1 = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                   kernel_initializer='he_uniform', name='conv1')(inputsDCE)
    conv1 = BatchNormalization(axis=-1, name='BN1')(conv1)
    conv1 = Activation(activation='relu', name='act_1')(conv1)
    
    conv1_2 = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                     kernel_initializer='he_uniform', name='conv1_2')(conv1)
    conv1_2 = BatchNormalization(axis=-1, name='BN1_2')(conv1_2)
    conv1_2 = Activation(activation='relu', name='act_1_2')(conv1_2)
    #block 2
    conv2 = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                   kernel_initializer='he_uniform', name='conv2')(conv1_2)
    conv2 = BatchNormalization(axis=-1, name='BN2')(conv2)
    conv2 = Activation(activation='relu', name='act_2')(conv2)
    
    conv2_2 = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                     kernel_initializer='he_uniform', name='conv2_2')(conv2)
    conv2_2 = BatchNormalization(axis=-1, name='BN2_2')(conv2_2)
    conv2_2 = Activation(activation='relu', name='act_2_2')(conv2_2)
    #block 3
    conv3 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                   kernel_initializer='he_uniform', name='conv3')(conv2_2)
    conv3 = BatchNormalization(axis=-1, name='BN3')(conv3)
    conv3 = Activation(activation='relu', name='act_3')(conv3)
    
    conv3_2 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                     kernel_initializer='he_uniform', name='conv3_2')(conv3)
    conv3_2 = BatchNormalization(axis=-1, name='BN3_2')(conv3_2)
    conv3_2 = Activation(activation='relu', name='act_3_2')(conv3_2)
    
    conv3_3 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                     kernel_initializer='he_uniform', name='conv3_3')(conv3_2)
    conv3_3 = BatchNormalization(axis=-1, name='BN3_3')(conv3_3)
    conv3_3 = Activation(activation='relu', name='act_3_3')(conv3_3)
    #block 4
    conv4 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                   kernel_initializer='he_uniform', name='conv4')(conv3_3)
    conv4 = BatchNormalization(axis=-1, name='BN4')(conv4)
    conv4 = Activation(activation='relu', name='act_4')(conv4)
    
    conv4_2 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                     kernel_initializer='he_uniform', name='conv4_2')(conv4)
    conv4_2 = BatchNormalization(axis=-1, name='BN4_2')(conv4_2)
    conv4_2 = Activation(activation='relu', name='act_4_2')(conv4_2)
    
    conv4_3 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                     kernel_initializer='he_uniform', name='conv4_3')(conv4_2)
    conv4_3 = BatchNormalization(axis=-1, name='BN4_3')(conv4_3)
    conv4_3 = Activation(activation='relu', name='act_4_3')(conv4_3)
    
    #block 5
    conv5 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                   kernel_initializer='he_uniform', name='conv5')(conv4_3)
    conv5 = BatchNormalization(axis=-1, name='BN5')(conv5)
    conv5 = Activation(activation='relu', name='act_5')(conv5)
    
    conv5_2 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                     kernel_initializer='he_uniform', name='conv5_2')(conv5)
    conv5_2 = BatchNormalization(axis=-1, name='BN5_2')(conv5_2)
    conv5_2 = Activation(activation='relu', name='act_5_2')(conv5_2)
    
    conv5_3 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                     kernel_initializer='he_uniform', name='conv5_3')(conv5_2)
    conv5_3 = BatchNormalization(axis=-1, name='BN5_3')(conv5_3)
    conv5_3 = Activation(activation='relu', name='act_5_3')(conv5_3)
    
    # DWI
    #block 1
    conv1D = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                    kernel_initializer='he_uniform', name='conv1D')(inputsDWI)
    conv1D = BatchNormalization(axis=-1, name='BN1D')(conv1D)
    conv1D = Activation(activation='relu', name='act_1D')(conv1D)
    
    conv1_2D = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                      kernel_initializer='he_uniform', name='conv1_2D')(conv1D)
    conv1_2D = BatchNormalization(axis=-1, name='BN1_2D')(conv1_2D)
    conv1_2D = Activation(activation='relu', name='act_1_2D')(conv1_2D)
    #block 2
    conv2D = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                    kernel_initializer='he_uniform', name='conv2D')(conv1_2D)
    conv2D = BatchNormalization(axis=-1, name='BN2D')(conv2D)
    conv2D = Activation(activation='relu', name='act_2D')(conv2D)
    
    conv2_2D = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                      kernel_initializer='he_uniform', name='conv2_2D')(conv2D)
    conv2_2D = BatchNormalization(axis=-1, name='BN2_2D')(conv2_2D)
    conv2_2D = Activation(activation='relu', name='act_2_2D')(conv2_2D)
    #block 3
    conv3D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                    kernel_initializer='he_uniform', name='conv3D')(conv2_2D)
    conv3D = BatchNormalization(axis=-1, name='BN3D')(conv3D)
    conv3D = Activation(activation='relu', name='act_3D')(conv3D)
    
    conv3_2D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                      kernel_initializer='he_uniform', name='conv3_2D')(conv3D)
    conv3_2D = BatchNormalization(axis=-1, name='BN3_2D')(conv3_2D)
    conv3_2D = Activation(activation='relu', name='act_3_2D')(conv3_2D)
    
    conv3_3D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                      kernel_initializer='he_uniform', name='conv3_3D')(conv3_2D)
    conv3_3D = BatchNormalization(axis=-1, name='BN3_3D')(conv3_3D)
    conv3_3D = Activation(activation='relu', name='act_3_3D')(conv3_3D)
    #block 4
    conv4D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                    kernel_initializer='he_uniform', name='conv4D')(conv3_3D)
    conv4D = BatchNormalization(axis=-1, name='BN4D')(conv4D)
    conv4D = Activation(activation='relu', name='act_4D')(conv4D)
    
    conv4_2D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                      kernel_initializer='he_uniform', name='conv4_2D')(conv4D)
    conv4_2D = BatchNormalization(axis=-1, name='BN4_2D')(conv4_2D)
    conv4_2D = Activation(activation='relu', name='act_4_2D')(conv4_2D)
    
    conv4_3D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                      kernel_initializer='he_uniform', name='conv4_3D')(conv4_2D)
    conv4_3D = BatchNormalization(axis=-1, name='BN4_3D')(conv4_3D)
    conv4_3D = Activation(activation='relu', name='act_4_3D')(conv4_3D)
    
    #block 5
    conv5D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                    kernel_initializer='he_uniform', name='conv5D')(conv4_3D)
    conv5D = BatchNormalization(axis=-1, name='BN5D')(conv5D)
    conv5D = Activation(activation='relu', name='act_5D')(conv5D)
    
    conv5_2D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                      kernel_initializer='he_uniform', name='conv5_2D')(conv5D)
    conv5_2D = BatchNormalization(axis=-1, name='BN5_2D')(conv5_2D)
    conv5_2D = Activation(activation='relu', name='act_5_2D')(conv5_2D)
    
    conv5_3D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                      kernel_initializer='he_uniform', name='conv5_3D')(conv5_2D)
    conv5_3D = BatchNormalization(axis=-1, name='BN5_3D')(conv5_3D)
    conv5_3D = Activation(activation='relu', name='act_5_3D')(conv5_3D)
    

    concat = layers.concatenate([conv1_2, conv1_2D, conv2_2, conv2_2D, conv3_2,
                                 conv3_2D, conv4_2, conv4_2D, conv5_2, conv5_2D], axis=concat_axis)
    
    #block 6
    dropout1 = Dropout(0.2)(concat)
    
    conv6 = Conv2D(128, (1,1), activation=None, dilation_rate=1,
                   kernel_initializer='he_uniform', name='conv6')(dropout1)
    conv6 = BatchNormalization(axis=-1, name='BN6')(conv6)
    conv6 = Activation(activation='relu', name='act_6')(conv6)
    
    dropout2 = Dropout(0.2)(conv6)

    
    conv6_2 = layers.Conv2D(2, (1, 1), activation='softmax')(dropout2)
    
    model = models.Model(inputs=[inputsDCE, inputsDWI], outputs=conv6_2)
    model.compile(optimizer=optimizers.Adam(lr=0.0001, decay=0.0),
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal') # lr was 0.001

    return model

# Data augmentation and random selection during training
def iterate_in_mb_train(train_dce, train_dwi, train_y):
    global batch_size
    
    while True:
        i_liver = np.random.choice(train_dce.shape[0], int(batch_size), replace=False)
        
        slices_input_dce = train_dce[i_liver[0],:,:,:]
        slices_input_dce = slices_input_dce[np.newaxis,:,:,:]
        
        slices_input_dwi = train_dwi[i_liver[0],:,:,:]
        slices_input_dwi = slices_input_dwi[np.newaxis,:,:,:]
        
        slices_target = train_y[i_liver[0],:,:]
        slices_target = slices_target[np.newaxis,:,:]

        lim_deg = 45 # limit of degrees of rotation
        deg = np.random.choice((range(-lim_deg,lim_deg)), int(batch_size))
        
        for p in range(1, int(batch_size)):
            p_input_dce = train_dce[i_liver[p],:,:,:]
            p_input_dwi = train_dwi[i_liver[p],:,:,:]
            p_target = train_y[i_liver[p],:,:]
            
            p_input_dce = sp.rotate(p_input_dce, deg[p], reshape=False, order=3)
            p_input_dwi = sp.rotate(p_input_dwi, deg[p], reshape=False, order=3)
            p_target = sp.rotate(p_target, deg[p], reshape=False, order=1)
                
            p_input_dce = p_input_dce[np.newaxis,:,:,:]
            p_input_dwi = p_input_dwi[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:]
            
            slices_input_dce = np.concatenate((slices_input_dce, p_input_dce), axis=0)
            slices_input_dwi = np.concatenate((slices_input_dwi, p_input_dwi), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)

        
            
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target[:,:,:,np.newaxis]
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target[:,:,:,np.newaxis]  # *5 times the weight, in this case 5.  
        
        yield [slices_input_dce, slices_input_dwi], mb_labels


def iterate_in_mb_test(test_dce, test_dwi, test_y):
    global batch_size
    
    while True:
        i_liver = np.random.choice(test_dce.shape[0], int(batch_size), replace=False)
        
        slices_input_dce = test_dce[i_liver[0],:,:,:]
        slices_input_dce = slices_input_dce[np.newaxis,:,:,:]
        
        slices_input_dwi = test_dwi[i_liver[0],:,:,:]
        slices_input_dwi = slices_input_dwi[np.newaxis,:,:,:]
        
        slices_target = test_y[i_liver[0],:,:]
        slices_target = slices_target[np.newaxis,:,:]
        
        lim_deg = 45 # limit of degrees of rotation
        deg = np.random.choice((range(-lim_deg,lim_deg)), int(batch_size))
        
        for p in range(1, int(batch_size)):
            p_input_dce = test_dce[i_liver[p],:,:,:]
            p_input_dwi = test_dwi[i_liver[p],:,:,:]
            p_target = test_y[i_liver[p],:,:]
            
            p_input_dce = sp.rotate(p_input_dce, deg[p], reshape=False, order=3)
            p_input_dwi = sp.rotate(p_input_dwi, deg[p], reshape=False, order=3)
            p_target = sp.rotate(p_target, deg[p], reshape=False, order=1)
            
                
            p_input_dce = p_input_dce[np.newaxis,:,:,:]
            p_input_dwi = p_input_dwi[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:]
            
            slices_input_dce = np.concatenate((slices_input_dce, p_input_dce), axis=0)
            slices_input_dwi = np.concatenate((slices_input_dwi, p_input_dwi), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)
        
            
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target[:,:,:,np.newaxis]
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target[:,:,:,np.newaxis] #*5  weighting of class 1 
        
        yield [slices_input_dce, slices_input_dwi], mb_labels



def bbox(img):
    img = np.sum(img, axis=0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(sitk_img, fname)



train_dce = []
train_dwi = []
train_y = []
test_dce = []
test_dwi = []
test_y = []

# Load the data and divide in patches
#basedir = r'/input/Data/DCEDWI/' #8
basedir = r'/input/Data/DCEDWI/' #5

for t in range(1,6):
    DCE, DWI, y = load_data_train(basedir, number=t)
        
    DCEaug = DCE[:, 50:178, 40:168, :]
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 60:188, :]),axis=0)
    
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 50:178, :]),axis=0)
    
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 40:168, :]),axis=0)
    
    DWIaug = DWI[:, 50:178, 40:168, :]
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 60:188, :]),axis=0)
    
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 50:178, :]),axis=0)
    
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 40:168, :]),axis=0)
        
    y4 = y[:, 50:178, 40:168]
    y4 = np.concatenate((y4, y[:, 30:158, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 60:188]),axis=0)
    
    y4 = np.concatenate((y4, y[:, 30:158, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 50:178]),axis=0)
    
    y4 = np.concatenate((y4, y[:, 50:178, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 40:168]),axis=0)
        
    if not np.any(train_dce):
        train_dce = DCEaug
        train_dwi = DWIaug
        train_y = y4
    else:
        train_dce = np.concatenate((train_dce, DCEaug), axis=0)
        train_dwi = np.concatenate((train_dwi, DWIaug), axis=0)
        train_y = np.concatenate((train_y, y4), axis=0)

    DCE, DWI, y = load_data_val(basedir, number=t)
            
    DCEaug = DCE[:, 50:178, 40:168, :]
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 60:188, :]),axis=0)
        
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 50:178, :]),axis=0)
        
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 20:148, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 30:158, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 50:178, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 50:178, 60:188, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 30:158, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 40:168, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 60:188, 40:168, :]),axis=0)
    DCEaug = np.concatenate((DCEaug, DCE[:, 70:198, 40:168, :]),axis=0)
        
    DWIaug = DWI[:, 50:178, 40:168, :]
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 60:188, :]),axis=0)
        
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 50:178, :]),axis=0)
        
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 20:148, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 30:158, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 50:178, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 50:178, 60:188, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 30:158, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 40:168, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 60:188, 40:168, :]),axis=0)
    DWIaug = np.concatenate((DWIaug, DWI[:, 70:198, 40:168, :]),axis=0)
            
    y4 = y[:, 50:178, 40:168]
    y4 = np.concatenate((y4, y[:, 30:158, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 60:188]),axis=0)
    
    y4 = np.concatenate((y4, y[:, 30:158, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 50:178]),axis=0)
    
    y4 = np.concatenate((y4, y[:, 50:178, 20:148]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 30:158]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 50:178]),axis=0)
    y4 = np.concatenate((y4, y[:, 50:178, 60:188]),axis=0)
    y4 = np.concatenate((y4, y[:, 30:158, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 40:168, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 60:188, 40:168]),axis=0)
    y4 = np.concatenate((y4, y[:, 70:198, 40:168]),axis=0)
            
    if not np.any(test_dce):
        test_dce = DCEaug
        test_dwi = DWIaug
        test_y = y4
    else:
        test_dce = np.concatenate((test_dce, DCEaug), axis=0)
        test_dwi = np.concatenate((test_dwi, DWIaug), axis=0)
        test_y = np.concatenate((test_y, y4), axis=0)
            
    if t == 5:
        print("All training and validation files loaded.")

train_dce = np.asarray(train_dce)
train_dwi = np.asarray(train_dwi)
train_y = np.asarray(train_y)
test_dce = np.asarray(test_dce)
test_dwi = np.asarray(test_dwi)
test_y = np.asarray(test_y)

n_classes = 2
batch_size = 4

# Weights per class
idx_1 = np.asarray(np.where(train_y==1))
idx_0 = np.asarray(np.where(train_y==0))
total = idx_0.shape[1]+idx_1.shape[1]
n_1 = idx_1.shape[1]/total
n_0 = idx_0.shape[1]/total
w_1 = 1/np.log(n_1+1.2)
w_0 = 1/np.log(n_0+1.2)
class_weights = np.array([w_0, w_1])

FinalModel = build_network(Inputshape1=(128, 128, 6), Inputshape2 =(128,128,3), num_class=2)


model_checkpoint = ModelCheckpoint('/input/lesionDetectionPnet_DCEDWI.h5',
                                   monitor='val_loss', verbose=0,
                                   save_best_only=True, mode='min')
 
tbCallback = TensorBoard(log_dir='/input/logs/lesionDetection_Pnet_DCEDWI',
                         histogram_freq=0, write_graph=True,
                         write_images=True)
n_epochs = 5000
FinalModel.fit_generator(iterate_in_mb_train(train_dce, train_dwi, train_y), 10,
                         epochs=n_epochs,
                         callbacks=[tbCallback,model_checkpoint], verbose=0,
                         validation_data=iterate_in_mb_test(test_dce, test_dwi, test_y),
                         validation_steps=5, class_weight=class_weights) 

#FinalModel.save_weights('/input/lesionDetectionPnet_DCEDWI.h5', overwrite=True)


## Testing step
basedir = r'C:\Users\user\Documents\Detection\Data\Testing'
LesionSegmentation = build_network(Inputshape1=(256, 256, 6), Inputshape2 =(256,256,3), num_class=2)

LesionSegmentation.load_weights('H:/Docker/lesionDetectionPnet_DCEDWI.h5')
for t in range(1,65):
    file = path.join(basedir, str(t),'e-THRIVE_reg.nii')

    
    if path.isfile(file):
       inputDCE, inputDWI, _ , _ = load_data(basedir, nameImage='e-THRIVE_reg.nii', number=t)

       test_DCE = inputDCE[0,:,:,:]
       test_DCE = test_DCE[np.newaxis, :, :, :]
       test_DWI = inputDWI[0,:,:,:]
       test_DWI = test_DWI[np.newaxis, :, :, :]
       predict_50 = LesionSegmentation.predict([test_DCE, test_DWI])
       predict_50 = predict_50[0,:,:,:]
       prediction = predict_50[:,:,1]*100
       prediction = prediction[np.newaxis,:,:]
           
       for slice in range(1,inputDCE.shape[0]):
           test_DCE = inputDCE[slice,:,:,:]
           test_DCE = test_DCE[np.newaxis, :, :, :]
           test_DWI = inputDWI[slice,:,:,:]
           test_DWI = test_DWI[np.newaxis, :, :, :]
           predict_50 = LesionSegmentation.predict([test_DCE, test_DWI])
           predict_50 = predict_50[0,:,:,:]
           predict_50 = predict_50[:,:,1]*100
           prediction = np.concatenate((prediction, predict_50[np.newaxis,:,:]), axis=0)
               
       prediction = np.asarray(prediction, dtype='int')

