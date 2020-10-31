from image_transform import ImageTransform
from my_dataset import HymenopteraDataset
from utils import MyUtils
import os

if __name__ == "__main__":

    phase = "train"

    target_files = MyUtils.make_datapath_list(phase)
    
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(size, mean, std)
    train_ds = HymenopteraDataset(target_files, transform, phase)

    print(train_ds)
    print(train_ds.__getitem__(0))
    print(train_ds.__getitem__(0)[0].size())
