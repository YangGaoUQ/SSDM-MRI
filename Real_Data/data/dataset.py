import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
from .mask import (Gen_Sampling_Mask,get_mask)
from cmb_multi import adaptive_cmb_2d,reconstruct_multi_channel

def convert_to_k_space(mri_image):

    k_space = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mri_image)))
    return k_space

def k_space_to_img(k_space):

    img = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k_space))))
    return img

def normalize_npy(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def make_dataset(dir):
    file_path = []
    assert os.path.isdir(dir), f'{dir} Invalid directory'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            file_path.append(path)
    return file_path



class MRI_Restoration(data.Dataset):
    def __init__(self, data_root,acc_factor=-1, mask_type=None,image_size=[320, 320]):
        self.acc_factor=acc_factor
        self.mask_type=mask_type
        self.paths = make_dataset(data_root)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.paths[index]
        img = np.load(path).astype(np.float32)

        if img.ndim == 2:  # 2D 数据
            img_kspace = convert_to_k_space(img)
            mask = get_mask(self.image_size,self.acc_factor,self.mask_type)
            under_kspace = img_kspace * mask
            cond_img = k_space_to_img(under_kspace).astype(np.float32)

            img = normalize_npy(img)
            cond_img = normalize_npy(cond_img)

            gt_img = img / np.max(img)
            k_full = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gt_img)))
            k_sub = k_full * mask.astype(np.float32)

            img = torch.from_numpy(img)
            img = torch.unsqueeze(img, dim=0)
            cond_img = torch.from_numpy(cond_img)
            cond_img = torch.unsqueeze(cond_img, dim=0)

            ret['gt_image'] = img
            ret['cond_image'] = cond_img
            ret['sub_kspace'] = k_sub
            ret['mask'] = mask

        #multi-coil data
        elif img.ndim == 3:
            img_all = []
            cond_img_all = []
            cond_ksp=[]
            mask = get_mask(self.image_size,self.acc_factor,self.mask_type)

            for i in range(img.shape[0]):
                img_slice = img[i]  # 取出第 i 个 2D 切片
                img_kspace = convert_to_k_space(img_slice)
                under_kspace = img_kspace * mask

                cond_img = k_space_to_img(under_kspace).astype(np.float32)

                gt_img = img_slice / np.max(img)
                k_full = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gt_img)))
                k_sub = k_full * mask.astype(np.float32)

                img_all.append(img_slice)
                cond_img_all.append(cond_img)
                cond_ksp.append(k_sub)


            img_all = np.stack(img_all, axis=0)
            cond_img_all = np.stack(cond_img_all, axis=0)
            img_combined,coil_sen_img=adaptive_cmb_2d(img_all)
            cond_img_combined, coil_sen_cond_img =adaptive_cmb_2d(cond_img_all)


            img_combined = normalize_npy(img_combined.astype(np.float32))
            img_combined = torch.from_numpy(img_combined)
            img_combined = torch.unsqueeze(img_combined, dim=0)
            cond_img_combined = normalize_npy(cond_img_combined.astype(np.float32))
            cond_img_combined = torch.from_numpy(cond_img_combined)
            cond_img_combined = torch.unsqueeze(cond_img_combined, dim=0)

            cond_ksp = np.array(cond_ksp)
            ret['gt_image'] = img_combined
            ret['cond_image'] = cond_img_combined
            ret['sub_kspace'] = cond_ksp
            ret['coil_sen']=coil_sen_cond_img
            ret['mask'] = mask

        else:
            raise ValueError(f"Unsupported data dimension: {img.ndim}. Expected 2D or 3D data.")

        # 提取文件名
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.paths)











