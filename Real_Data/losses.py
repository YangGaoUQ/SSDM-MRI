import torch.nn.functional as F
import torch
def discriminator_loss(discriminator, v, v_t):
    real_loss = F.mse_loss(discriminator(v_t), torch.ones_like(discriminator(v_t)))
    fake_loss = F.mse_loss(discriminator(v), torch.zeros_like(discriminator(v)))
    return 0.5 * (real_loss + fake_loss)

def generator_loss(discriminator, v):
     return  F.mse_loss(discriminator(v), torch.ones_like(discriminator(v)))