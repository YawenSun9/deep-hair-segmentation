import os
import numpy as np
import scipy.misc as m
from PIL import Image
from mypath import Path
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders import joint_transforms as jnt_trnsf
import torchvision.transforms as std_trnsf

class LFWSegmentation(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self, args, root=Path.db_root_dir('lfw'), img_size = 256, split="train"):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): True if to add gray images
        """
        self.split = split
        self.args = args
        
        if self.split == 'train':
            txt_file = 'parts_train.txt'
        elif self.split == 'val':
            txt_file = 'parts_validation.txt'
        elif self.split == 'test':
            txt_file = 'parts_test.txt'
        
        txt_dir = os.path.join(root, txt_file)
        name_list = LFWSegmentation.parse_name_list(txt_dir)
        img_dir = os.path.join(root, 'lfw_funneled')
        mask_dir = os.path.join(root, 'parts_lfw_funneled_gt_images')

        self.img_path_list = [os.path.join(img_dir, elem[0], elem[1]+'.jpg') for elem in name_list]
        self.mask_path_list = [os.path.join(mask_dir, elem[1]+'.ppm') for elem in name_list]
        
        if self.split == 'train':
            self.joint_transforms, self.image_transforms, self.mask_transforms = self.train_transform(img_size)
        elif self.split == 'val':
            self.joint_transforms, self.image_transforms, self.mask_transforms = self.val_transform(img_size)
        elif self.split == 'test':
            self.joint_transforms, self.image_transforms, self.mask_transforms = self.test_transform(img_size)
        

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)
        mask = LFWSegmentation.rgb2binary(mask)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

#         if self.gray_image:
#             gray = img.convert('L')
#             gray = np.array(gray,dtype=np.float32)[np.newaxis,]/255
#             return img, mask, gray
        _, M, N = mask.shape
        sample = {'image': img, 'label': mask.resize_((M, N))}
#         print('lfw', sample['image'].shape, sample['label'].shape)
        return sample
    

    def __len__(self):
        return len(self.mask_path_list)
    
    def train_transform(self, img_size):
        # transforms on both image and mask
        train_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.RandomCrop(self.args.crop_size),
        jnt_trnsf.RandomRotate(5),
        jnt_trnsf.RandomHorizontallyFlip()
        ])

        # transforms only on images
        train_image_transforms = std_trnsf.Compose([
#         jnt_trnsf.RandomGaussianBlur(),
        std_trnsf.ColorJitter(0.05, 0.05, 0.05, 0.05),
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        # transforms only on mask
        mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])
        
        return train_joint_transforms, train_image_transforms, mask_transforms
    
    def val_transform(self, img_size):
        val_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.FixScaleCrop(self.args.crop_size)
        ])

        val_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # transforms only on mask
        mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])
        
        return val_joint_transforms, val_image_transforms, mask_transforms

    def test_transform(self, img_size):
        test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.FixedResize(self.args.crop_size)
        ])

        test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # transforms only on mask
        mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])
        
        return test_joint_transforms, test_image_transforms, mask_transforms
    

    @staticmethod
    def rgb2binary(mask):
        """transforms RGB mask image to binary hair mask image.
        """
        mask_arr = np.array(mask)
        mask_map = mask_arr == np.array([255, 0, 0])
        mask_map = np.all(mask_map, axis=2).astype(np.float32)
        return Image.fromarray(mask_map)

    @staticmethod
    def parse_name_list(fp):
        with open(fp, 'r') as fin:
            lines = fin.readlines()
        parsed = list()
        for line in lines:
            name, num = line.strip().split(' ')
            num = format(num, '0>4')
            filename = '{}_{}'.format(name, num)
            parsed.append((name, filename))
        return parsed
