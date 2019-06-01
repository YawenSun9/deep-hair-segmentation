import argparse
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
import os
import numpy as np
import matplotlib.pyplot as plt
from webcam import norm
from dataloaders.utils import decode_seg_map_sequence


def get_gradient(image, logit, target):
    sobel_kernel_x = torch.Tensor(
                [[1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0],
                [1.0, 0.0, -1.0]])

    sobel_kernel_x = sobel_kernel_x.view((1,1,3,3))
    
    N = target.shape[0]
    
    I_x1 = F.conv2d(logit[:,0:1,...], sobel_kernel_x, padding = 1)
    I_x2 = F.conv2d(logit[:,1:2,...], sobel_kernel_x, padding = 1)
    I_x = torch.cat([I_x1, I_x2], dim=1)
    M_x = F.conv2d(target, sobel_kernel_x, padding = 1)

    sobel_kernel_y = torch.Tensor(
                [[1.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -1.0]])

    sobel_kernel_y = sobel_kernel_y.view((1,1,3,3))

    I_y1 = F.conv2d(logit[:,0:1,...], sobel_kernel_y, padding = 1)
    I_y2 = F.conv2d(logit[:,1:2,...], sobel_kernel_y, padding = 1)
    I_y = torch.cat([I_y1, I_y2], dim=1)
    M_y = F.conv2d(target, sobel_kernel_y, padding = 1)

    for i in range(8):
        plt.imshow(image[i].permute(1,2,0))
        plt.show()
        plt.imshow(I_x1[i,0,...], cmap='gray')
        plt.show()
        plt.imshow(I_y1[i,0,...], cmap='gray')
        plt.show()
        plt.imshow(I_x2[i,0,...], cmap='gray')
        plt.show()
        plt.imshow(I_y2[i,0,...], cmap='gray')
        plt.show()
        plt.imshow(target[0,0,...], cmap='gray')
        plt.show()
        plt.imshow(M_x[i,0,...], cmap='gray')
        plt.show()
        plt.imshow(M_y[i,0,...], cmap='gray')
        plt.show()


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # print(lam)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    
def show_mixup(original_image, image, targets_a, targets_b, lam, output, dataset):
    # for j in range(args.batch_size):
    #     plt.imshow(image[j].permute(1,2,0))
    #     plt.show()
    # image = norm(image.permute(0, 2,3,1)).permute(0,3,1,2)

    mask = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset)
    for j in range(args.batch_size):
        plt.imshow(image[j].permute(1,2,0))
        plt.show()
        plt.imshow(mask[j].permute(1,2,0))
        plt.show()
        # print(targets_a.shape)
        # print(targets_b[j]*(1 - lam).shape)
        plt.imshow(targets_a[j]*lam + targets_b[j]*(1 - lam), cmap='gray')
        plt.show()
    for j in range(args.batch_size):
        plt.imshow(original_image[j].permute(1,2,0))
        plt.show()


def test(args):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=False)
    model.load_state_dict(torch.load(args.pretrained, map_location=device)['state_dict'])
    model.eval()
    tbar = tqdm(test_loader) ## train test dev
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        # original_image = image
        if args.use_mixup:
            image, targets_a, targets_b, lam = mixup_data(image, target,
                                                          args.mixup_alpha, use_cuda=False)
        # mixed_image = image
        # image = norm(image.permute(0,2,3,1)).permute(0,3,1,2)
        output = model(image)
        # if lam > 0.4 and lam < 0.6:
        #     show_mixup(original_image, mixed_image, targets_a, targets_b, lam, output, args.dataset)
        # get_gradient(image, output.detach(), target.unsqueeze(dim=1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="webcam")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--pretrained', type=str, 
                        default='/Users/yulian/Downloads/mixup_model_best.pth.tar',
                        help='pretrained model')
    parser.add_argument('--dataset', type=str, default='lfw',
                        choices=['pascal', 'lfw', 'cityscapes', 'celebA', 'lfw_celebA'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=227,
                        help='crop image size')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--use-mixup', type=bool, default=False,
                        help='whether to include data mixup (default: False)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        metavar='M', help='alpha (default: 0.2)')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(args)
