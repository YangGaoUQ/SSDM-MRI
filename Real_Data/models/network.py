import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from .guided_diffusion_modules.unet import UNet
import os
from data.dataset import convert_to_k_space,k_space_to_img
from cmb_multi import reconstruct_multi_channel,adaptive_cmb_2d
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def data_consistency(y, sub_kspace, mask, device, coil_sen=None, coeff1=0.5):
    y = y.cpu().numpy()
    mask = mask.numpy()
    sub_kspace = sub_kspace.numpy()

    y = y[0, 0, :, :]
    mask = mask[0, :, :]
    mask = mask.astype(np.float32)

    if sub_kspace.ndim == 3:
        sub_kspace = sub_kspace[0, :, :]
        k_sub = sub_kspace
        rec_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(y)))

        max_k_sub = np.max(np.abs(k_sub))
        max_rec_k = np.max(np.abs(rec_k))

        rec_k = rec_k / max_rec_k * max_k_sub
        rec_k_dc = rec_k * (1 - mask) + k_sub*mask*coeff1+rec_k*mask*(1-coeff1)

        y_dc = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(rec_k_dc)))


    elif sub_kspace.ndim == 4:
        coil_sen = coil_sen.numpy()
        sub_kspace = sub_kspace[0, :, :, :]
        coil_sen = coil_sen[0, :, :, :]

        y_multi=reconstruct_multi_channel(y,coil_sen)

        for i in range(y_multi.shape[2]):
            k_sub=sub_kspace[i,:,:]
            multi_slice=y_multi[:,:,i]
            rec_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(multi_slice)))

            max_k_sub = np.max(np.abs(k_sub))
            max_rec_k = np.max(np.abs(rec_k))

            rec_k = rec_k / max_rec_k * max_k_sub
            rec_k_dc = rec_k * (1 - mask) + k_sub*mask*coeff1+rec_k*mask*(1-coeff1)
            multi_slice_dc = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(rec_k_dc)))
            y_multi[:,:,i]=multi_slice_dc
        y_dc,sen=adaptive_cmb_2d(img = np.transpose(y_multi, (2,1,0)))
        y_dc_flipped = np.fliplr(y_dc)
        y_dc = np.rot90(y_dc_flipped, k=1)

    y_dc = y_dc / np.max(np.abs(y_dc))
    y_dc = np.expand_dims(np.expand_dims(y_dc, axis=0), axis=0)
    y_dc = torch.from_numpy(y_dc).float()  # 转换为 PyTorch 张量
    y_dc = y_dc.to(device)  # 将张量移动到指定设备（例如GPU）
    return y_dc




class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3',time_scale=None,distill=False,**kwargs):
        super(Network, self).__init__(**kwargs)
        if distill:
            self.denoise_fn=unet
        else:
            self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.time_scale=time_scale
        self.nsteps=None

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        #alphas累乘得到gammas
        gammas = np.cumprod(alphas, axis=0)

        gammas_prev = np.append(1., gammas[:-1])

        # 扩散 Q（x_t | x_{t-1}） 等的计算
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.gammas)
        # 后验 Q（x_{t-1} | x_t， x_0） 的计算
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: 由于扩散链开头的后验方差为 0，因此对数计算被裁剪
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    # def predict_start_from_noise(self, y_t, t, noise):
    #     return (
    #         extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
    #         extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
    #     )

    #v-diffusion
    def predict_start_from_noise(self, y_t, t, noise):
        alpha=extract(torch.sqrt(self.gammas), t, y_t.shape)
        sigma=extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        return  (y_t * alpha - noise * sigma)

    #后验分布中的均值和方差
    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None,mask=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance



    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None,mask=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond,mask=mask)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None,mask=None, sample_num=8,sub_kspace=None,coil_sen=None):
        b, *_ = y_cond.shape
        assert self.num_timesteps >= sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond,mask=mask)
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        if mask!=None:
            y_t = data_consistency(y_t,sub_kspace,mask,device=y_cond.device,coil_sen=coil_sen)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        #sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t.long() - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t.long(), x_shape=(1, 1))

        #进行线性插值，生成一组样本的 gammas 参数，这个参数用于后续的采样
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))

        # 使用采样得到的 gammas 参数进行信号采样，得到带有噪声的信号
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        # 使用采样得到的 gammas 参数对带噪声信号进行去噪，得到去噪后的噪声
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)

        alpha = extract(torch.sqrt(self.gammas), t, y_cond.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, y_cond.shape)
        v=alpha * noise - sigma * y_0
        loss = self.loss_fn(v, noise_hat)
        return loss

# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

#张量 a 中根据索引 t 提取值，并确保提取的张量形状与给定的 x_shape 相匹配。这样可以用于在扩散模型中获取对应时间步的参数
def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule功能
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas




