from scipy.fft import fftn, ifftn
from numpy.linalg import svd
import numpy as np




def adaptive_cmb_2d(img, vox=[1, 1, 1], cref=1, radi=5):
    img = np.transpose(img, (1, 2, 0))
    img = np.expand_dims(img, axis=-1)

    npix, nv, nrcvrs, ne = img.shape
    img_orig = img.copy()
    img = img[..., 0]  # 取第一个回波

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
    # return img_cmb, sen.reshape(npix, nv, nrcvrs)
def reconstruct_multi_channel(img_cmb, sen):
    """
    利用线圈灵敏度图将2D切片重建为多通道数据。
    参数:
        img_cmb (numpy.ndarray): 合成的2D切片，形状为 [H, W]。
        sen (numpy.ndarray): 线圈灵敏度图，形状为 [H, W, nc]，其中 nc 是通道数。

    返回:
        multi_recon (numpy.ndarray): 重建的多通道数据，形状为 [H, W, nc]。
    """
    # 检查输入形状
    if img_cmb.ndim != 2:
        raise ValueError("img_cmb 必须是2D数组，形状为 [H, W]")
    if sen.ndim != 3:
        raise ValueError("sen 必须是3D数组，形状为 [H, W, nc]")
    if img_cmb.shape != sen.shape[:2]:
        raise ValueError("img_cmb 和 sen 的前两个维度必须一致")

    H, W = img_cmb.shape
    nc = sen.shape[2]
    multi_recon = np.zeros((H, W, nc), dtype=np.complex64)

    for c in range(nc):
        multi_recon[:, :, c] = img_cmb * sen[:, :, c]
    return multi_recon


def fft2c(img):
    """2D 频率变换，仿 MATLAB `fft2`"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    """2D 逆频率变换，仿 MATLAB `ifft2`"""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))




# mat_data = sio.loadmat('under_img.mat')
#
# # 取出数据
# under_img = mat_data['under_img']  # 假设存储键名是 'under_img'
# print("MAT 文件加载完成，形状：", under_img.shape)
#
# # 确保数据类型为复数（MATLAB 可能存成 float 类型的实部和虚部）
# if not np.iscomplexobj(under_img) and 'under_img_imag' in mat_data:
#     under_img = under_img + 1j * mat_data['under_img_imag']
#
# # 运行函数
# img_cmb, sen = adaptive_cmb_2d(under_img)
#
# # 打印组合图像和灵敏度图的形状
# print("Combined image shape:", img_cmb.shape)
# print("Coil sensitivity shape:", sen.shape)
#
# # 调用 reconstruct_multi_channel 函数
# multi_recon = reconstruct_multi_channel(img_cmb[..., 0], sen)  # 使用第一个回波的组合图像
# print("Reconstructed multi-channel data shape:", multi_recon.shape)
#
# # 可视化重建的多通道数据
# n_channels = multi_recon.shape[2]  # 获取通道数
# n_rows = 4  # 每行显示4个通道
# n_cols = n_channels // n_rows  # 计算需要的列数
#
# fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20, 16))  # 每行显示4个通道，每通道显示幅度和相位
# fig.suptitle('Reconstructed Multi-channel Data', fontsize=16)
#
# # 显示每个通道的幅度和相位
# for c in range(n_channels):
#     row = (c // n_cols) * 2  # 计算幅度和相位的行索引
#     col = c % n_cols  # 计算列索引
#
#     # 幅度
#     axes[row, col].imshow(np.abs(multi_recon[:, :, c]), cmap='gray')
#     axes[row, col].set_title(f'Channel {c+1} Magnitude')
#     axes[row, col].axis('off')
#
#     # 相位
#     axes[row + 1, col].imshow(np.angle(multi_recon[:, :, c]), cmap='gray')
#     axes[row + 1, col].set_title(f'Channel {c+1} Phase')
#     axes[row + 1, col].axis('off')
#
# plt.tight_layout()
# plt.show()