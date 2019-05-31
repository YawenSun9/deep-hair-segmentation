import numpy as np
import pydensecrf.densecrf as dcrf
import torch

def dense_crf(img, output_probs):
    '''
    argv:
    img: nparray; has to be unit8
    output_probs: tensor; raw output of net; has to be float32
    
    return: 
    nparray int64
    '''
    output_probs = torch.from_numpy(np.array(output_probs))
    
    h = output_probs.shape[0] # 576
    w = output_probs.shape[1] # 864

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
    U = -np.log(output_probs)
    
    U = U.astype('float32')
    # print(U.shape)        # -> (2, 576, 864)
    # print(U.dtype)        # -> dtype('float32'), has to be float32
    
    U = U.reshape((2, -1))
    # print(U.shape) # (2, 497664)
    
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    
    return Q # nparray int64
'''
# demo
from PIL import Image
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

img = Image.open('toy.jpg')
mask = Image.open('toymask.jpg')

mask = np.array(mask)
img = np.array(img).astype('uint8')
mask_crf = dense_crf(img, mask)
# print(mask_crf.shape, type(mask_crf), mask_crf.dtype)
# mask_crf = Image.fromarray(mask_crf.astype('uint8')) # change to image to show
%matplotlib inline
plt.imshow(mask_crf)
plt.show()
'''