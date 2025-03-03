import os
import sys
import time
import datetime
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image

from helmet_detr import build_helmet_detr, HELMET_CLASSES

class HelmetDataset(Dataset):
    """
    Dataset for Helmet Detection.
    Expects annotations in COCO format with 5 classes:
    1. rider_with_helmet
    2. rider_without_helmet
    3. rider_and_passenger_with_helmet
    4. rider_and_passenger_without_helmet
    5. rider_with_helmet_and_passenger_without_helmet
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create image id to annotations mapping
        self.img_to_ann = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_ann:
                self.img_to_ann[img_id] = []
            self.img_to_ann[img_id].append(ann)
        
        # Create a list of image ids that have annotations
        self.img_ids = list(self.img_to_ann.keys())
        
        # Create a mapping from image id to image info
        self.id_to_img = {img['id']: img for img in self.coco['images']}
        
        # Create a mapping from category id to our internal class id (0-4)
        self.cat_to_class = {}
        for i, cat in enumerate(self.coco['categories']):
            self.cat_to_class[cat['id']] = i
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Get image id
        img_id = self.img_ids[idx]
        
        # Get image info
        img_info = self.id_to_img[img_id]
        
        # Get image path
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.img_to_ann[img_id]
        
        # Create target dictionary
        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }
        
        # Process annotations
        for ann in anns:
            # Get bounding box
            x, y, w, h = ann['bbox']
            
            # Handle empty boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to [x_center, y_center, width, height] format and normalize
            x_c = (x + w / 2) / img_info['width']
            y_c = (y + h / 2) / img_info['height']
            w = w / img_info['width']
            h = h / img_info['height']
            
            # Add to target
            target['boxes'].append([x_c, y_c, w, h])
            target['labels'].append(self.cat_to_class[ann['category_id']])
        
        # Convert to tensors
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.long)
        
        # Apply transforms if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


class RandomHorizontalFlip:
    """
    Randomly flip images and targets horizontally.
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = F.hflip(image)
            
            # Flip boxes
            boxes = target['boxes']
            boxes[:, 0] = 1 - boxes[:, 0]  # Flip center x
            target['boxes'] = boxes
        
        return image, target


class ToTensor:
    """
    Convert PIL image to tensor and normalize.
    """
    def __call__(self, image, target):
        # Convert image to tensor
        image = F.to_tensor(image)
        
        # Normalize
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, target


class Compose:
    """
    Compose several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    # Stack images
    images = torch.stack(images)
    
    return images, targets


def get_transforms(train=True):
    """
    Get transforms for training or validation.
    """
    transforms = []
    
    if train:
        # Add data augmentation for training
        transforms.append(RandomHorizontalFlip(0.5))
    
    # Add conversion to tensor for all datasets
    transforms.append(ToTensor())
    
    return Compose(transforms)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, print_freq=10):
    """
    Train the model for one epoch.
    """
    model.train()
    criterion.train()
    
    # Statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        data_time.update(time.time() - start_