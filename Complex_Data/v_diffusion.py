import math
CUDA_LAUNCH_BLOCKING=1
import torch
import torch.nn.functional as F
from inspect import isfunction
from functools import partial


def make_diffusion(model, n_timestep, time_scale, device):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
    return GaussianDiffusion(model, betas, time_scale=time_scale)


def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise Exception()
    return betas


def E_(input, t, shape=(1,1,1,1)):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




class GaussianDiffusion:

    def __init__(self, net, betas, time_scale=1, sampler="ddpm"):
        super().__init__()
        self.net_ = net
        self.time_scale = time_scale
        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64, device=betas.device), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)



    def inference(self, x, y_noise, gammas):
        return self.net_(torch.cat([x, y_noise], dim=1).float(), gammas)


    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma


class GaussianDiffusionDefault(GaussianDiffusion):

    def __init__(self, net, betas, time_scale=1, gamma=0.3):
        super.__init__(net, betas, time_scale)
        self.gamma = gamma

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def distill_loss(self, student_diffusion, gt_x, x, t, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(gt_x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1) #[1,1,1,1]
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)

            gammas=extract(self.alphas_cumprod,t+1,x_shape=(1,1))#[1,1]
            gammas_1=extract(self.alphas_cumprod,t,x_shape=(1,1))
            gammas_s=extract(student_diffusion.alphas_cumprod,t//2,gt_x)

            sqrt_recip_gammas=extract(torch.sqrt(1. / self.alphas_cumprod),t+1,gt_x.shape)
            sqrt_recipm1_gammas=extract(torch.sqrt_(1. / self.alphas_cumprod - 1),t+1,gt_x.shape)

            sqrt_recip_gammas_1 = extract(torch.sqrt(1. / self.alphas_cumprod), t , gt_x.shape)
            sqrt_recipm1_gammas_1 = extract(torch.sqrt_(1. / self.alphas_cumprod - 1), t , gt_x.shape)

            z = alpha * gt_x + sigma * eps
            v = self.inference(x,z,gammas).double()
            rec=sqrt_recip_gammas*z-sqrt_recipm1_gammas*v.clip(-1,1)

            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
            v_1 = self.inference(x,z_1,gammas_1).double()
            x_2 = (sqrt_recip_gammas_1 * z_1 - sqrt_recipm1_gammas_1 * v_1).clip(-1, 1)

            eps_2 = (z - alpha_s * x_2) / sigma_s  #eps = (z - alpha*x)/sigma
            v_2 = alpha_s * eps_2 - sigma_s * x_2

            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)

        v = student_diffusion.net_(torch.cat([x, z], dim=1).float(), gammas_s)
        return F.mse_loss(w * v.float(), w * v_2.float())




