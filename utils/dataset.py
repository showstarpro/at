import os
import sys
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import os.path
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image

def find_inputs(root, filename_to_target=None):
    inputs = []
    for filename, target in filename_to_target.items():
        abs_filename = os.path.join(root, filename)
        inputs.append((abs_filename, target))

    return inputs

class  Dataset(data.Dataset):
    def __init__(self, root, target_file, transform=None):
        super().__init__()

        if target_file:
            target_file_path = target_file #os.path.join(root, target_file)
            target_df = pd.read_csv(target_file_path)
            target_df["label"] = target_df["label"].apply(int)
            f_to_t = dict(zip(target_df["filename"], target_df["label"] - 1)) # -1 for 0-999 class ids
        else:
            f_to_t = dict()
        
        imgs = find_inputs(root=root, filename_to_target=f_to_t)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
    

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).log
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    def set_transform(self, transform):
        self.transform = transform
    
    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]
            
        



