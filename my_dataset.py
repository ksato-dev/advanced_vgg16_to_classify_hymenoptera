### ハチとアリを分類するためのデータセット（torchのDatasetベースで作る）
import torch
import torch.utils.data as data

from PIL import Image

class HymenopteraDataset(data.Dataset):
    """
    ハチとアリを分類するためのデータセット（torchのDatasetベースで作る）

    Attributes:
    ----------------------
    _file_list: image paths
    _transform: function to preprocess image
    _phase: "train" or "val"
    ----------------------
    """

    def __init__(self, file_list, transform = None, phase = "train"):
        self._file_list = file_list
        self._transform = transform
        self._phase = phase
    
    def __len__(self):
        """
        return length of images
        """

        return len(self._file_list)

    def __getitem__(self, index):
        """
        return a transformed image that specified index, and a label either hymenoptera or ant
        """
        image_path = self._file_list[index]
        input_image = Image.open(image_path)

        # ret_image = self._transform[phase](input_image)
        ret_image = self._transform(input_image, self._phase)

        # extract label from source-file
        ret_label = None
        if (self._phase == "train"):
            ret_label = image_path[30:34]
        elif (self._phase == "val"):
            ret_label = image_path[28:32]

        if (ret_label == "ants"):
            ret_label = 0
        elif (ret_label == "bees"):
            ret_label = 1
        
        return ret_image, ret_label