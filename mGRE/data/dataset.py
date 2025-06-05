import torch.utils.data as data
import os
import torch
from .util.mask import (custom_uniform_downsampling_mask,Gen_Sampling_Mask,get_mask)
import numpy as np
import random
import matplotlib.pyplot as plt

def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def complex_array_to_abs(input_array):
    tensor_input = torch.from_numpy(input_array)
    if tensor_input.ndim == 4:
        real_part = tensor_input[:, 0, :, :]
        imag_part = tensor_input[:, 1, :, :]
    elif tensor_input.ndim == 3:
        real_part = tensor_input[0,:, :]
        imag_part = tensor_input[1,:, :]
    img = real_part + 1j * imag_part
    result_image = np.abs(img)
    return result_image.numpy(),img.numpy()

def k_img(kspace):
    img = np.zeros_like(kspace, dtype=np.complex128)
    for i in range(kspace.shape[3]):
        img[:, :, :, i] = np.fft.fftn(np.fft.fftshift(kspace[:, :, :, i].real+1j*kspace[:,:,:,i].imag))
    return img

def fft2c(x):
    res = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))
    return res

def ifft2c(x):
    res = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    return res

def make_dataset(dir):
    file_path=[]
    assert os.path.isdir(dir), f'{dir} 不是一个有效的目录'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            file_path.append(path)
    return file_path


class MyDataSet(data.Dataset):
    def __init__(self, data_root,acc_factor=-1,image_size=[256, 256],AF=-1):
        self.paths= make_dataset(data_root)
        self.acc_factor = acc_factor
        self.paths = make_dataset(data_root)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.paths[index]
        img = np.load(path)
        img_kspace=fft2c(img)
        mask=Gen_Sampling_Mask(img_kspace.shape,self.acc_factor)
        under_kspace=img_kspace * mask
        cond_image=ifft2c(under_kspace)

        k_full = fft2c(img)
        k_sub = k_full * mask.astype(np.float32)

        norm_img = img / np.max(np.abs(img))
        norm_cond_image = cond_image / np.max(np.abs(cond_image))

        img_real = torch.from_numpy(np.real(norm_img))
        img_real = torch.unsqueeze(img_real, dim=0)
        img_imag = torch.from_numpy(np.imag(norm_img))
        img_imag = torch.unsqueeze(img_imag, dim=0)

        cond_image_real=torch.from_numpy(np.real(norm_cond_image))
        cond_image_real=torch.unsqueeze(cond_image_real, dim=0)
        cond_image_imag=torch.from_numpy(np.imag(norm_cond_image))
        cond_image_imag=torch.unsqueeze(cond_image_imag,dim=0)


        gt_image = torch.cat([img_real, img_imag], dim=0)
        cond_image=torch.cat([cond_image_real,cond_image_imag], dim=0)
        gt_image = gt_image.to(dtype=torch.float32)
        cond_image = cond_image.to(dtype=torch.float32)

        ret['gt_image'] = gt_image
        ret['cond_image'] = cond_image
        ret['mask']=mask
        ret['norm_max'] = np.max(np.abs(img))
        ret['sub_kspace'] = k_sub

        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]

        return ret

    def __len__(self):
        return len(self.paths)








