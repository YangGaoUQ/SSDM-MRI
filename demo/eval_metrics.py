import numpy as np
import math
from skimage.metrics import structural_similarity as ssim


def generate_mask(img, threshold=None):
    mask = img > threshold
    return mask.astype(np.float32)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def compute_metrics(gt_img, recon_img, threshold=0.03):
    mask = generate_mask(gt_img, threshold)
    gt_img = gt_img * mask
    recon_img = recon_img * mask

    return calculate_psnr(gt_img, recon_img), calculate_ssim(gt_img, recon_img)