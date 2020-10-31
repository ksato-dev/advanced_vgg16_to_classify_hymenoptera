from image_transform import ImageTransform
from my_dataset import HymenopteraDataset
from utils import MyUtils
import os
from tqdm import tqdm

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

def get_params():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return size, mean, std

def test(phase = "train"):
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

def train(train_data_loader, val_data_loader, criterion, optimizer, num_epochs, net):

    loaders = {"train":train_data_loader, "val":val_data_loader}

    for curr_epoch in tqdm(range(num_epochs)):
        
        for phase in ["train", "val"]:

            if (phase == "train"):
                net.train()
            elif (phase == "val"):
                net.eval()

            if (curr_epoch == 0 and phase == "train"):
                continue

            curr_loader = loaders[phase]
            total_loss = 0.0
            num_corrects = 0

            print(phase, curr_loader)

            for inputs, labels in curr_loader:                    
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    curr_loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  ## axis(row) = 1 に対する最大値, 確率

                    if (phase == "train"):
                        curr_loss.backward()
                        optimizer.step()

                    total_loss += curr_loss.item() * inputs.size(0)
                    num_corrects += torch.sum(preds == labels.data)

                    # print("preds")
                    # print(preds)
                    # print("labels.data")
                    # print(labels.data)
                    # print("-------------")

            total_loss = total_loss / len(loaders[phase].dataset)
            total_acc = num_corrects.double() / len(loaders[phase].dataset)
                
            print("Phase:{}, Loss:{:.4f}, Acc:{:.4f}".format(phase, total_loss, total_acc))
                
    pass

def get_dataset(phase):

    target_files = MyUtils.make_datapath_list(phase)
    size, mean, std = get_params()
    transform = ImageTransform(size, mean, std)
    ret_dataset = HymenopteraDataset(target_files, transform, phase)

    return ret_dataset

def main():
    data_loaders = {}
    batchsize = 50
    for phase in ["train", "val"]:
        enable_shuffle = True
        if (phase == "val"):
            enable_shuffle = False

        data_loaders[phase] = torch_data.DataLoader(
            get_dataset(phase), batch_size = batchsize, shuffle = enable_shuffle)
    
    criterion = nn.CrossEntropyLoss()

    vgg_net = models.vgg16(pretrained=True)
    
    vgg_net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)

    keys_to_change = ["classifier.6.weight", "classifier.6.bias"]
    params_to_update = MyUtils.get_specified_params_from_torch_net(vgg_net, keys_to_change)
    optimizer = optim.SGD(params = params_to_update, lr = 0.001, momentum = 0.9)

    num_epochs = 2
    train(data_loaders["train"], data_loaders["val"], criterion, optimizer, num_epochs, vgg_net)

if __name__ == "__main__":

    # test("train")
    # test("val")
    # print(train_ds)
    # print(train_ds.__getitem__(0))
    # print(train_ds.__getitem__(0)[0].size())

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main()
