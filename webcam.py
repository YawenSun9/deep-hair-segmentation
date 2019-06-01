import argparse
import cv2
import torch
from modeling.deeplab import DeepLab
import os
import numpy as np


def norm(image):
    return (image - torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 1, 1, -1)) / \
           torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 1, 1, -1)

def fix_ratio_scale(image, size):
    image_h, image_w = image.shape[0], image.shape[1]
    if image_h < image_w:
        resize = image_h
        dw = int((image_w - image_h) / 2)
        image = image[:, dw : dw + image_h, :]
    else:
        resize = image_w
        dh = int((image_h - image_w) / 2)
        image = image[dh: dh + image_w, ...]
    
    return image, resize

def get_image_mask(image, net, size = 227):
    image_h, image_w = image.shape[0], image.shape[1]

    image, resize = fix_ratio_scale(image, size)
    down_size_image = cv2.resize(image, (size, size))
    b, g, r = cv2.split(down_size_image)
    down_size_image = cv2.merge([r,g,b])
    down_size_image = torch.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    down_size_image = norm(down_size_image)
    down_size_image = np.transpose(down_size_image, (0, 3, 1, 2)).to(device)
    mask = net(down_size_image)

    mask = torch.argmax(torch.squeeze(mask), 0)
    mask_cv2 = mask.data.cpu().numpy().astype(np.uint8) * 255
    mask_cv2 = cv2.resize(mask_cv2, (resize, resize))
    return image, mask_cv2


def color_image(image, mask, color):
    c = None
    if color == 'purple':
        c = [30, 0, 10]
    elif color == 'green':
        c = [5, 20, 5]
    elif color == 'blue':
        c = [30, 5, 0]
    elif color == 'red':
        c = [10, 10, 30]
    hand = np.zeros((mask.shape[0], mask.shape[1], 3))
    hand[np.where(mask != 0)] = c
    hand = hand.astype(np.uint8)
    return cv2.add(hand, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="webcam")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--pretrained', type=str, 
                        default='/Users/yulian/Downloads/resnet_model_best.pth.tar',
                        help='pretrained model')
    parser.add_argument('--color', type=str, default='purple',
                        choices=['purple', 'green', 'blue', 'red'],
                        help='Color your hair (default: purple)')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DeepLab(backbone=args.backbone, output_stride=16, num_classes=2, sync_bn=False).to(device)
    net.load_state_dict(torch.load(args.pretrained, map_location=device)['state_dict'])
    net.eval()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("webcam is not detected")
 
    while (True):
        # ret : frame capture(boolean)
        # frame : Capture frame
        ret, image = cam.read()

        if (ret):
            image, mask = get_image_mask(image, net)
            # print(image.shape, mask.shape)
            add = color_image(image, mask, args.color)
            cv2.imshow('frame', add)
            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break

    cam.release()
    cv2.destroyAllWindows()