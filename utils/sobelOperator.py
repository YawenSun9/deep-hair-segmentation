import numpy as np
from matplotlib import pyplot as plt 
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

    argvs: img(tensor) 2d
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
        self.img = img

    def x_only_mask(self):
        G_x = F.conv2d(self.img, self.sobel_kernel_x, stride=1, padding=1)
        return G_x
    def y_only_mask(self):
        G_y = F.conv2d(self.img, self.sobel_kernel_y, stride=1, padding=1)
        return G_y
    def x_y_mask(self):
        G_x = F.conv2d(self.img, self.sobel_kernel_x, stride=1, padding=1)
        G_y = F.conv2d(self.img, self.sobel_kernel_y, stride=1, padding=1)
        G_xy = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        return G_xy

'''
# demo

if __name__ == "__main__":
    # read file
    img = cv2.cvtColor(cv2.imread("sol_key.png"), cv2.COLOR_BGR2GRAY) # GRAYs
    H, W = img.shape
    img = torch.from_numpy(img).float().view((1,1,H,W))
    
    # print(img.shape)
    # apply sobel
    img1 = Sobel(img).x_only_mask()
    img2 = Sobel(img).y_only_mask()
    img3 = Sobel(img).x_y_mask()
    
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
    
    # visualize
    plt.subplot(3,1,1); plt.imshow(img1.view((H,W)));plt.axis('off');plt.title('sobel_x_only')
    plt.subplot(3,1,2); plt.imshow(img2.view((H,W)));plt.axis('off');plt.title('sobel_y_only')
    plt.subplot(3,1,3); plt.imshow(img3.view((H,W)));plt.axis('off');plt.title('sobel_x_y')
    plt.show()

    # output file
'''