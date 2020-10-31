from image_transform import ImageTransform
from my_dataset import HymenopteraDataset
from utils import MyUtils
import os
from tqdm import tqdm

import torch.utils.data as torch_data

def get_params():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return size, mean, std

def test():
    phase = "train"

    target_files = MyUtils.make_datapath_list(phase)
    
    size, mean, std = get_params()

    transform = ImageTransform(size, mean, std)
    train_ds = HymenopteraDataset(target_files, transform, phase)

    # print(train_ds)
    # print(train_ds.__getitem__(0))
    # print(train_ds.__getitem__(0)[0].size())

    ## use data loader
    train_loader = torch_data.DataLoader(
        train_ds, batch_size = 80, shuffle = True)
    
    # print()
    # print(train_loader)
    train_iter = iter(train_loader)
    for inputs, labels in tqdm(train_iter):
        print(inputs.size())
        print(labels.size())

    print()
    print(train_ds.__len__())

if __name__ == "__main__":
    test()

    pass