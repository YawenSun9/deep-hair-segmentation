import torch
import Sobel
class gradientConsistancyLoss():
    def __init__(self):
        super(gradientConsistancyLoss, self).__init__()
    
    def forward(self, input, target):
        return gradient_consistancy_loss(input, target)

def gradient_consistancy_loss(input, target):
    loss = None
    # input image
    I_x = Sobel(input).x_only_mask() # i am not sure the dim of input and target
    I_y = Sobel(input).y_only_mask()

    # mask
    M_x = Sobel(target).x_only_mask()
    M_y = Sobel(target).y_only_mask()
    
    M_Mag = Sobel(target).x_y_mask()
    rang_grad = 1 - torch.pow(I_x*M_x + I_y*M_y,2)
    
    loss_a = M_Mag*rang_grad
    loss_b = M_Mag # I am not sure for the sum here
    return loss_a, loss_b