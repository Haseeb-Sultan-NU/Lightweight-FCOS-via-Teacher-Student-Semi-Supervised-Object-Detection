import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class DSLCOCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, transforms=None):
        """
        Args:
            img_dir (str): Directory containing the raw images.
            annotation_file (str): Path to the COCO JSON file.
            transforms (callable, optional): Weak or Strong augmentations.
        """
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load the massive JSON into memory once
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create quick lookup dictionaries for speed
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image ID so the GPU can fetch them instantly
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Fetch the image
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Open image and convert to standard RGB
        image = Image.open(img_path).convert("RGB")
        
        # 2. Fetch the labels
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO format is [x_min, y_min, width, height]
            x_min, y_min, w, h = ann['bbox']
            
            # MATH: Convert to [Center_X, Center_Y, Width, Height] for FCOS+
            cx = x_min + (w / 2)
            cy = y_min + (h / 2)
            
            boxes.append([cx, cy, w, h])
            labels.append(ann['category_id'])

        # Convert to PyTorch Tensors (The language the GPU speaks)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        # 3. Apply the Gym (Augmentations) if any exist
        if self.transforms:
            # PyTorch v2 transforms handle both the image and the boxes simultaneously
            image, target = self.transforms(image, target)

        return image, target