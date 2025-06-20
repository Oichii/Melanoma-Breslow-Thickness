from torch.utils.data import Dataset
import os
import cv2
import albumentations as albu
import torch
import numpy as np


class ThicknessDataset(Dataset):
    def __init__(self, dataframe, root_dir='', img_path='', classification=False, transforms=None, ext='.jpg',
                 test=False, tta=False):
        self.df = dataframe
        self.img_path = os.path.join(root_dir, img_path)
        self.classification = classification

        self.ext = ext
        self.test = test
        self.tta = tta

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = albu.Compose([
                albu.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        if self.test:
            image = cv2.imread(os.path.join(self.img_path, self.df.iloc[item, 0]+self.ext))
        else:
            image = cv2.imread(os.path.join(self.img_path, self.df.iloc[item, 0]+self.ext))
        # print(os.path.join(self.img_path, self.df.iloc[item, 0]+self.ext))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.tta:
            return image, self.df.iloc[item, 3], self.df.iloc[item, 4]

        image = self.transforms(image=image)
        image = image['image'].astype(np.float32)

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).float()

        if self.classification:
            label = self.df.iloc[item, 3]
            label = torch.tensor(label, dtype=torch.long)  # categorical label
        else:
            label = self.df.iloc[item, 1]
            label = torch.tensor(label, dtype=torch.float)  # continuous label

        return image, label, self.df.iloc[item, 4], self.df.iloc[item, 0]


def get_transforms(image_size, mean, std):
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    transforms_train = albu.Compose([
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.5),
        albu.Resize(image_size, image_size),
        albu.Transpose(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # albu.OneOf([
        #     albu.MotionBlur(blur_limit=5),
        #     albu.MedianBlur(blur_limit=5),
        #     albu.GaussianBlur(blur_limit=(3, 5), sigma_limit=1),
        #     albu.GaussNoise(var_limit=(5.0, 30.0)),
        # ], p=0.7),

        # albu.OneOf([
        #     albu.OpticalDistortion(distort_limit=1.0),
        #     albu.GridDistortion(num_steps=5, distort_limit=1.),
        #     # albu.ElasticTransform(alpha=3),
        # ], p=0.7),

        # albu.CLAHE(clip_limit=4.0, p=0.7),
        # albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

        albu.RandomRotate90(p=0.5),
        # albu.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.7),
        albu.Normalize()
    ])

    transforms_val = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize()
    ])

    return transforms_train, transforms_val

# 0.253, 0.347, 0.654
# 0.243, 0.114, 0.113
