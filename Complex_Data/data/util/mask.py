import torch
import numpy as np
import  math
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def Gen_Sampling_Mask(MatrixSize,AF=-1):
    # pa = 7, pb = 1.8AF = 2;(0.5 sampling),
    # pa = 12, pb = 1.8, AF = 4;(0.25 sampling),
    # pa = 17, pb = 1.8 AF = 6;
    # pa = 22, pb = 1.8, AF = 8;

    if AF==4:
        Pa=12
        Pb=1.8

    elif AF==5:
        Pa=27
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