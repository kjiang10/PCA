from numpy.lib.npyio import load
import torch
from torch.utils.data.sampler import BatchSampler
from prepare_data import format_file_name, get_celebA_x, get_celebA_ya, vectorize_image
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils import data
from create_shift import create_shift, load_data_xay
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        # print(self.data)
        print(len(self.data))
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_xya = self.data[index]
        data_x, data_yz = data_xya
        data_x = self.transform(data_x)
        return [data_x, data_yz]

def format_xya(idx, num_samples, y, a):
    data_x = get_celebA_x(idx, num_samples)
    # pick 3rd column as label and 5th column as sensitive attribute
    data_ya = get_celebA_ya(y, a, idx, num_samples)
    src = []
    # print(len(data_x), len(data_ya))
    for i in range(len(idx)):
        src.append([data_x[i], data_ya[i]])
    return src

def prepare(num_samples, y, a):
    data = vectorize_image(num_samples)
    # create_shift should only be called once
    src_idx, tgt_idx, _ = create_shift(data, src_split = .4, sampling = 'pca', mean_a = 0 , std_b = 1, kdebw = .3)

    # print((set(src_idx) & set(tgt_idx)))
    # print(src_idx)
    # print(tgt_idx)
    return format_xya(src_idx, num_samples, y, a), format_xya(tgt_idx, num_samples, y, a)
    
    # trainloader = torch.utils.data.DataLoader(src, shuffle=False, batch_size=1)
    # i1, l1 = next(iter(trainloader))
    # print(i1.shape)

def get_loader():
    transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    # fetch examples from the dataset and perform PCA to split the data
    src_data, tgt_data = prepare(num_samples=2000, y=3, a=5)

    src_data = CelebADataset(data=src_data, transform=transform)
    tgt_data = CelebADataset(data=tgt_data, transform=transform)
    src_loader = DataLoader(src_data, batch_size=50, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_data, batch_size=50, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    # for batch_idx, batch in enumerate(src_loader):
    #     print(batch_idx, batch[0].shape)
    # for batch_idx, batch in enumerate(tgt_loader):
    #     print(batch_idx, batch[0].shape)

    return src_loader, tgt_loader
