from matplotlib import image
import pandas as pd
import numpy as np
import glob
import os
from PIL import Image

import torch
import torchvision
import matplotlib as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils import calculate_gcc

class phenocamdata(torch.utils.data.Dataset):
    def __init__(self, image_path, site_name, image_height, image_width, roi_id, file_type, is_train = True, data_vol=1):
        super().__init__()
        self.is_train = is_train
        self.data_vol = data_vol
        if is_train:
            self.image_path = os.path.join(image_path, "train", site_name)
        else:
            self.image_path = os.path.join(image_path, "test", site_name)
        self.roi_path = os.path.join(image_path, "roi")
        self.image_height = image_height
        self.image_width = image_width
        self.site_name = site_name
        self.roi_id = roi_id
        self.file_type = file_type
        self.roi_stats_url = "https://phenocam.sr.unh.edu/data/archive/"+self.site_name+"/ROI/"+self.site_name+"_"+self.roi_id+"_roistats.csv"
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        file = os.path.join(row["image_file_path"], row["image_file_name"])
        return (
            self.preprocess(Image.open(file).convert('RGB')), row["gcc_rounded"], row["image_file_name"])
    
    def create_image_dataframe(self):
        roi_stats_df = pd.read_csv(self.roi_stats_url, header = 17)
        roi_stats_df.dropna(subset = ["gcc"], inplace=True)
        image_dict = {"image_file_name":[],"image_file_path":[]}
        for (root, subdirs, files) in os.walk(self.image_path):
            for subdir in subdirs:
                for (root, subsubdirs, files) in os.walk(os.path.join(self.image_path,subdir)):
                    for subsubdir in subsubdirs:
                        img_path = os.path.join(root, subsubdir)
                        img_files = glob.glob(img_path + self.file_type)
                        for file in img_files:
                            image_dict["image_file_name"].append(os.path.basename(file))
                            image_dict["image_file_path"].append(img_path)

        image_df = pd.DataFrame(image_dict)
        image_df = pd.merge(left = image_df, right = roi_stats_df[["filename", "gcc", "rcc"]], how='inner', 
                   left_on=['image_file_name'], right_on=['filename'])
        image_df["gcc_rounded"] = round(image_df["gcc"]*100, 2) 
        image_df.drop("filename", axis =1, inplace=True)
        del roi_stats_df
        if self.data_vol!=1:
            image_df = image_df.sample(frac=self.data_vol)
        return image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('RGB')
        return image
    
    def load_roi(self):
        roi_file_name = self.site_name+"_"+self.roi_id+"_01.tif"
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(((self.image_height, self.image_width))),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        roi = preprocess(Image.open(os.path.join(self.roi_path, roi_file_name)))
        # Replace white pixels with -1 and black pixels with 1
        roi = torch.where(roi==1, torch.Tensor([-1]), roi)
        roi = torch.where(roi==0, torch.Tensor([1]), roi)
        roi = torch.where(roi==-1, torch.Tensor([0]), roi)
        return roi
    
    
