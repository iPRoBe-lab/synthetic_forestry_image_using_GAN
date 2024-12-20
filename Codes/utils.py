import numpy as np
import os
import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_skimage
from pytorch_msssim import ssim, ms_ssim


def make_folder(path, folder_name):
        if not os.path.exists(os.path.join(path, folder_name)):
            os.makedirs(os.path.join(path, folder_name))

def denorm(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0, 1)

def tensor_to_PIL(img):
    img = transforms.functional.to_pil_image(img)
    return img

def calculate_gcc(img, roi):
    roi = torch.where(roi==-1, torch.Tensor([0]), roi)
    masked_img = img * roi
    r = torch.sum(masked_img[0, :, :])/torch.count_nonzero(roi)
    g = torch.sum(masked_img[1, :, :])/torch.count_nonzero(roi)
    b = torch.sum(masked_img[2, :, :])/torch.count_nonzero(roi)
    gcc = g/(r+g+b) 
    return gcc
    
def calculate_rcc(img,roi):
    roi = torch.where(roi==-1, torch.Tensor([0]), roi)
    masked_img = img * roi
    r = torch.sum(masked_img[0, :, :])/torch.count_nonzero(roi)
    g = torch.sum(masked_img[1, :, :])/torch.count_nonzero(roi)
    b = torch.sum(masked_img[2, :, :])/torch.count_nonzero(roi)
    rcc =  r/(r+g+b)
    return rcc
    
def calculate_ssim_score(img1, img2):
    ssim_score = ssim_skimage(img1, img2, channel_axis=2, data_range = img2.max() - img2.min()) 
    return ssim_score

def calculate_ssim(X,Y):
    ssim_val = ssim(X, Y, data_range=1, size_average=False)
    return ssim_val
    
def calculate_ms_ssim(X,Y):    
    ms_ssim_val = ms_ssim(X, Y, data_range=1, size_average=False)
    return ms_ssim_val
    