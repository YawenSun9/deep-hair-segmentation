import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F


''' 
ref: 
https://www.aiuai.cn/aifarm365.html
ct2.imread() # read file
'''

class Sobel:
    '''
    The sobel operator help us get the edges of an image
    At each point the result is either the corresponding gradient vector or the norm2 of this vector
    Convoluting the img on x and y derection 

    argvs: img(tensor) 2d ----not sure----
    '''
    def __init__(self, img):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sobel_kernel_x = torch.Tensor( # torch.Tensor
                    [[1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0]]).to(self.device).view((1,1,3,3))
        self.sobel_kernel_y = torch.Tensor(
                    [[1.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, -2.0, -1.0]]).to(self.device).view((1,1,3,3))
        self.H = img.shape[0]
        self.W = img.shape[1]
        self.img = img.view((1,1,self.H,self.W))

    def x_only_mask(self):
        G_x = F.conv2d(self.img, self.sobel_kernel_x, stride=1, padding=1)
        return G_x
    def y_only_mask(self):
        G_y = F.conv2d(self.img, self.sobel_kernel_y, stride=1, padding=1)
        return G_y
    def x_y_mask(self):
        G_x = F.conv2d(self.img, self.sobel_kernel_x, stride=1, padding=1)
        G_y = F.conv2d(self.img, self.sobel_kernel_y, stride=1, padding=1)
        G_xy = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2)).clamp(max=255)
        return G_xy

'''
# demo

if __name__ == "__main__":
    img = np.asarray(Image.open('sol_key.png').convert('L')).astype('float')
    H, W = img.shape
    img = torch.from_numpy(img).float()
    img = Sobel(img).x_y_mask()
    img = Image.fromarray(img.numpy().reshape((H,W)).astype('uint8'))
    plt.imshow(img)
    plt.show()
    
    
    # # it is total the same as using Image, but the output is wierd
    # img = cv2.cvtColor(cv2.imread("p.jpg"), cv2.COLOR_BGR2GRAY) # read file and to GRAYs image 
    # H, W = img.shape
    # img = torch.from_numpy(img).float().view((1,1,H,W)) # to tensor
    # img = Sobel(img).x_y_mask() # apply sobel
    # img = cv2.cvtColor(img.numpy().reshape((H,W)), cv2.COLOR_GRAY2RGB) # to image
    # plt.imshow(img)
    # plt.show()
'''

