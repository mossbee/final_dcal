"""
Data augmentation and transforms for FGVC and ReID tasks.

Provides task-specific data augmentation strategies following the paper specifications.
"""

import torchvision.transforms as T
import torch
import random
import math


def get_transforms(task='fgvc', split='train', img_size=448, crop_size=448):
    """
    Get appropriate transforms for a given task and split.
    
    FGVC Training:
        - Resize to 550x550
        - Random crop to 448x448
        - Random horizontal flip
        - Color jitter (optional)
        - ToTensor
        - Normalize (ImageNet mean/std)
    
    FGVC Testing:
        - Resize and center crop to 448x448
        - ToTensor
        - Normalize
    
    ReID Training (Person: 256x128, Vehicle: 256x256):
        - Resize
        - Random horizontal flip
        - Padding + random crop
        - Random erasing
        - ToTensor
        - Normalize
    
    ReID Testing:
        - Resize
        - ToTensor
        - Normalize
    
    Args:
        task (str): 'fgvc' or 'reid'
        split (str): 'train' or 'test'
        img_size (int or tuple): Target image size
        crop_size (int or tuple): Random crop size (for FGVC)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    # ImageNet normalization
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if task == 'fgvc':
        if split == 'train':
            # FGVC training: resize to 550, random crop to 448
            resize_size = int(crop_size * 550 / 448)
            transform = T.Compose([
                T.Resize(resize_size),
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            # FGVC testing: resize and center crop to 448
            transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                normalize
            ])
    
    elif task == 'reid':
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        if split == 'train':
            # ReID training with random erasing
            transform = T.Compose([
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.Pad(10),
                T.RandomCrop(img_size),
                T.ToTensor(),
                normalize,
                RandomErasing(probability=0.5)
            ])
        else:
            # ReID testing
            transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                normalize
            ])
    
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return transform


def get_train_transforms(task='fgvc', img_size=448, crop_size=448):
    """
    Get training transforms for a given task.
    
    Args:
        task (str): 'fgvc' or 'reid'
        img_size (int or tuple): Target image size
        crop_size (int or tuple): Random crop size (for FGVC)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms for training
    """
    return get_transforms(task=task, split='train', img_size=img_size, crop_size=crop_size)


def get_val_transforms(task='fgvc', img_size=448, crop_size=448):
    """
    Get validation/test transforms for a given task.
    
    Args:
        task (str): 'fgvc' or 'reid'
        img_size (int or tuple): Target image size
        crop_size (int or tuple): Center crop size (for FGVC)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms for validation/testing
    """
    return get_transforms(task=task, split='test', img_size=img_size, crop_size=crop_size)


class RandomErasing:
    """
    Random erasing augmentation for ReID.
    
    Randomly erases a rectangular region in the image to improve robustness
    and prevent overfitting.
    
    Args:
        probability (float): Probability of applying erasing
        sl (float): Min erasing area ratio
        sh (float): Max erasing area ratio
        r1 (float): Min aspect ratio
        mean (tuple): Mean values for erasing (per channel)
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        """Apply random erasing to image."""
        if random.uniform(0, 1) >= self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        
        return img

