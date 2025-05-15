import torch
import numpy as np
import  math
import os
from sigpy.mri import poisson
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_mask(img_size,acc_factor,type,center_fraction=None,fix=False,):
    if(acc_factor==-1):
        acc_factor = random.choice([4, 6, 8, 10, 12, 15])
    if(type is None):
        random_choice = random.randint(1, 4)
        if random_choice == 1:
            type = 'gaussian2d'
        elif random_choice == 2:
            type = 'gaussian1d'
        elif random_choice == 3:
            type = 'uniform1d'
        elif random_choice == 4:
            type = 'poisson'

    if acc_factor==4:
        center_fraction=0.08
    elif acc_factor==8 or acc_factor==6:
        center_fraction=0.04
    elif acc_factor==12 or acc_factor==10 or acc_factor==15:
        center_fraction=0.02

    size = img_size[0]  #Assuming img_size is a tuple (height, width)
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'gaussian2d':
        mask = torch.zeros(size, size)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[int_samples[:, 0], int_samples[:, 1]] = 1

    elif type == 'gaussian1d':
        mask = torch.zeros(size, size)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[c_from:c_from + Nsamp_center] = 1
        else:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[:, int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[:, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros(size, size)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[c_from:c_from + Nsamp_center] = 1
        else:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[:, samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[:, c_from:c_from+Nsamp_center] = 1
    elif type == 'poisson':
        mask = poisson((size, size), accel=acc_factor)
        mask = torch.from_numpy(mask)
        mask=np.real(mask)
    else:
        raise NotImplementedError(f'Mask type {type} is currently not supported.')
    return mask.numpy()


def Gen_Sampling_Mask(MatrixSize,AF=-1):
    # pa = 7, pb = 1.8AF = 2;(0.5 sampling),
    # pa = 12, pb = 1.8, AF = 4;(0.25 sampling),
    # pa = 17, pb = 1.8 AF = 6;
    # pa = 22, pb = 1.8, AF = 8;

    if AF==4:
        Pa=12
        Pb=1.8
    elif AF==6:
        Pa=17
        Pb=1.8

    elif AF==8:
        Pa=22
        Pb=1.8

    elif AF==10:
        Pa=27
        Pb=1.8

    elif AF==12:
        Pa=32
        Pb=1.8

    else :
        AF = 4
        Pa = 12
        Pb = 1.8

    nx, ny = MatrixSize
    x = np.arange(-nx//2, nx//2)
    y = np.arange(-ny//2, ny//2)
    Y, X = np.meshgrid(y, x)

    SP = np.exp(-Pa * ((np.sqrt(X**2 / nx**2 + Y**2 / ny**2))**Pb))
    SP_normalized = SP / (np.sum(SP) / (nx * ny / AF))
    mask = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if np.random.rand() < SP_normalized[i, j]:
                mask[i, j] = 1
    return mask
