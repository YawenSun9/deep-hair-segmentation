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

def get_image_mask(image, net, size = 257):
    mean = np.mean(image)
    image = cv2.flip(image, 1)
    image_h, image_w = image.shape[0], image.shape[1]
    image, resize = fix_ratio_scale(image, size)
    down_size_image = cv2.resize(image, (size, size))
    b, g, r = cv2.split(down_size_image)
    down_size_image = cv2.merge([r,g,b])
    down_size_image = torch.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    down_size_image = norm(down_size_image)
    down_size_image = np.transpose(down_size_image, (0, 3, 1, 2)).to(device)
    mask = net(down_size_image)
    mask = torch.squeeze(mask).detach()
    
    maskexp = np.exp(mask)
    prob = (maskexp[1] *255 / (maskexp[0] + maskexp[1] + 1e-30)).cpu().numpy().astype(np.int16)

    mask = torch.argmax(mask, 0)
    mask_cv2 = mask.data.cpu().numpy().astype(np.int16)
    mask_cv2 = cv2.resize(mask_cv2, (resize, resize))
    prob = cv2.resize(prob,  (resize, resize))
    # prob = prob * mean / 100
    return image, mask_cv2, prob


def color_image(image, mask, prob, color):
    c = None
    if color == 'purple':
        c = [30, 0, 10]
    elif color == 'grass':
        c = [0, 25, 20]
    elif color == 'green':
        c = [5, 20, 5]
    elif color == 'blue':
        c = [30, 5, 0]
    elif color == 'red':
        c = [10, 10, 30]
    elif color == 'rose':
        c = [15, 5, 30]
    # elif color == 'blond':
    #     c = [10, 25, 28]
    hand = np.zeros((mask.shape[0], mask.shape[1], 3))
    hand[np.where(mask != 0)] = c
    hand = hand * prob[...,np.newaxis] /255
    hand = hand.astype(np.uint8)
    return cv2.add(hand, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="webcam")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--pretrained', type=str, 
                        default='model/mobilewebcam_model_best.pth.tar',
                        help='pretrained model')
    parser.add_argument('--color', type=str, default='purple',
                        choices=['purple', 'green', 'blue', 'red', 'rose', 'grass'],
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
            image, mask, prob = get_image_mask(image, net)
            # print(image.shape, mask.shape)
            add = color_image(image, mask, prob, args.color)
            cv2.imshow('frame', add)
            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break

    cam.release()
    cv2.destroyAllWindows()
