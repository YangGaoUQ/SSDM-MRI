from scipy.fft import fftn, ifftn
from numpy.linalg import svd
import numpy as np

def adaptive_cmb_2d(img, vox=[1, 1, 1], cref=1, radi=5):
    img = np.transpose(img, (1, 2, 0))
    img = np.expand_dims(img, axis=-1)

    npix, nv, nrcvrs, ne = img.shape
    img_orig = img.copy()
    img = img[..., 0]

    kx = round(radi / vox[0])
    ky = round(radi / vox[1])
    kern = np.ones((kx * 2 + 1, ky * 2 + 1))

    tmp1 = np.tile(img.reshape(-1, nrcvrs)[:, :, np.newaxis], (1, 1, nrcvrs))  # (65536,16,16)
    tmp2 = np.tile(np.conj(img.reshape(-1, nrcvrs))[:, np.newaxis, :], (1, nrcvrs, 1))  # (65536,16,16)
    R = tmp1 * tmp2
    R = R.reshape(npix, nv, nrcvrs, nrcvrs)

    cvsize = (npix + 2 * kx + 1, nv + 2 * ky + 1)
    R_fft = fftn(R, s=cvsize, axes=(0, 1)) * fftn(kern, s=cvsize)[:, :, None, None]
    RS = ifftn(R_fft, axes=(0, 1)).real
    RS = RS[kx:npix + kx, ky:nv + ky]
    RS = RS.transpose(2, 3, 0, 1).reshape(nrcvrs, nrcvrs, npix * nv)

    sen = np.zeros((nrcvrs, npix * nv), dtype=complex)
    for i in range(RS.shape[2]):
        V, _, _ = svd(RS[:, :, i])
        sen[:, i] = V[:, 0]

    sen = sen.T.reshape(npix, nv, nrcvrs)
    sen /= np.expand_dims(sen[:, :, cref - 1] / np.abs(sen[:, :, cref - 1]), axis=2)

    img_cmb = np.zeros((npix, nv, ne), dtype=complex)
    sen = sen.reshape(-1, nrcvrs)

    for echo in range(ne):
        img_e = img_orig[..., echo].reshape(-1, nrcvrs)
        combined = np.sum(np.conj(sen) * img_e, axis=1) / np.sum(sen * np.conj(sen), axis=1)
        combined = combined.reshape(npix, nv)
        combined[np.isnan(combined)] = 0
        combined[np.isinf(combined)] = 0
        img_cmb[..., echo] = combined

    return img_cmb[:,:,0], sen.reshape(npix, nv, nrcvrs)
def reconstruct_multi_channel(img_cmb, sen):

    if img_cmb.ndim != 2:
        raise ValueError("img_cmb Must be a 2D array of shapes [H, W]")
    if sen.ndim != 3:
        raise ValueError("sen Must be a 3D array of shapes [H, W, nc]")
    if img_cmb.shape != sen.shape[:2]:
        raise ValueError("img_cmb and sen,the first two dimensions must be consistent")

    H, W = img_cmb.shape
    nc = sen.shape[2]
    multi_recon = np.zeros((H, W, nc), dtype=np.complex64)

    for c in range(nc):
        multi_recon[:, :, c] = img_cmb * sen[:, :, c]
    return multi_recon


def fft2c(img):

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):

    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))


