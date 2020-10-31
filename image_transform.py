import torch
import torchvision
from torchvision import transforms

class ImageTransform:
    """
    preprocessing images for deep learning.
    input: resize, mean, std 
    output: function to preprocess 
    """    
    def __init__(self, resize, mean, std):
        self._data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  ## convert to tensor
                transforms.Normalize(mean, std)]
                ),
            "val": transforms.Compose([
                transforms.Resize(resize),  
                transforms.CenterCrop(resize),  
                transforms.ToTensor(),  ## convert to tensor
                transforms.Normalize(mean, std)]
                )
        }

    def __call__(self, img, phase = "train"):
        """
        input: img, phase
            phase: "train" or "val"
        output: transformed img
        """
        return self._data_transform[phase](img)


