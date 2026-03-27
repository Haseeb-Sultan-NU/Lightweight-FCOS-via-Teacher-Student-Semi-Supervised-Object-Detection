import torch
from torchvision.transforms import v2

def get_weak_augmentations(train=True):
    """
    Used for the Teacher (Oracle) to generate clean Pseudo-Labels.
    Keeps the image mostly original, just simple flips and resizing.
    """
    transforms = []
    # Convert PIL Image to PyTorch Tensor
    transforms.append(v2.ToImage())
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    
    if train:
        # 50% chance to flip the image horizontally
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
        
    return v2.Compose(transforms)

def get_strong_augmentations():
    """
    Used for the Student (FCOS+). 
    Aggressive distortions force the CNN to learn deep feature representation 
    instead of just memorizing pixels.
    """
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        
        # Color Jitter: Randomly mess with brightness, contrast, and saturation
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        
        # Random Erasing (Acts like 'Cutout' from the DSL paper)
        # Drops black boxes over random parts of the image so the AI has to 
        # guess what the object is using only partial information.
        v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ]
    return v2.Compose(transforms)