import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'hair':
            return self.HairMatLoss
        else:
            raise NotImplementedError

    def HairMatLoss(self, logit, target, weight=0.5):
        loss1 = self.CrossEntropyLoss(logit, target)
        loss2 = weight * self.GradientConsistencyLoss(logit, target)
        print (loss1, loss2)
        return  loss1


    def GradientConsistencyLoss(self, logit, target):
        sobel_kernel_x = torch.Tensor(
                    [[1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0]])
        if self.cuda:
            sobel_kernel_x = sobel_kernel_x.cuda()
        sobel_kernel_x = sobel_kernel_x.view((1,1,3,3))
        N, H, W = target.shape

        I_x = F.conv2d(logit[:,1:2, ...], sobel_kernel_x, padding = 1)
        M_x = F.conv2d(target.view(N,1,H,W), sobel_kernel_x, padding = 1)

        sobel_kernel_y = torch.Tensor(
                    [[1.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, -2.0, -1.0]])
        if self.cuda:
            sobel_kernel_y = sobel_kernel_y.cuda()
        sobel_kernel_y = sobel_kernel_y.view((1,1,3,3))

        I_y = F.conv2d(logit[:,1:2, ...], sobel_kernel_y, padding = 1)
        M_y = F.conv2d(target.view(N,1,H,W), sobel_kernel_y, padding = 1)

        Imag_pow = torch.pow(I_x,2) + torch.pow(I_y,2) + 1e-30
        Mmag_pow = torch.pow(M_x,2) + torch.pow(M_y,2) + 1e-30
        Mmag = torch.sqrt(Mmag_pow)

        rang_grad = 1 - torch.pow(I_x*M_x + I_y*M_y,2) / Imag_pow / Mmag_pow
        loss = torch.sum(torch.mul(Mmag, rang_grad))/torch.sum(Mmag)

        if self.batch_average:
            loss /= N
        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




