
import os
import glob
import torch
from torchvision import models

class MyUtils:
    
    @staticmethod
    def get_specified_params_from_torch_net(torch_net, keys):
        ret_params = []
        
        for name, param in torch_net.named_parameters():
            if name in keys:
                param.requires_grad = True
                ret_params.append(param)
                print(name)
            else:
                param.requires_grad = False   ## keys にないパラメータは更新しない

        return ret_params

    @staticmethod
    def make_datapath_list(phase = "train"):
        """
        Inputs:
            - phase: "train" or "val"

        Returns:
            - file_list: a file_list that specified phase 
        """
        dataset_dir = "/home/ksato/dev/pytorch_advanced/1_image_classification/data/hymenoptera_data/"
        target_paths = os.path.join(dataset_dir + phase + "/**/*.jpg")
        print(target_paths)

        ret_file_list = []
        for curr_file in glob.glob(target_paths):
            ret_file_list.append(curr_file)

        return ret_file_list 

