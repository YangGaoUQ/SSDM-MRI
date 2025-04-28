"""
神经网络的各种实用程序。
"""

import math
import numpy as np
import torch 
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zeros out the module's parameters and returns.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Zoom the module's parameters and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the average of all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
   Create a standard normalization layer.

:p Number of ARAM Channels: The number of input channels.
    :return: a nn. Modules for normalization.
    """
    return GroupNorm32(32, channels)



def checkpoint(func, inputs, params, flag):
    """
    Evaluate the function without caching intermediate activations, thus allowing
    Reduced memory, but at the cost of additional computation in backward passing.

:p aram func: The function to be calculated.
    :p aram inputs: the sequence of arguments passed to 'func'.
    :p aram params: The parameter "func" sequence depends on but does not depend on
                   Explicitly as a parameter.
    :p aram flag: If false, gradient checkpointing is disabled.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():

            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_flops_attn(model, _x, y):
    """
    “thop”Counters for packages, which are used for calculations
    Pay attention to the operation.
    Designed to be used in the following ways:
        macs，parameter= thop.profile（

            inputs=（inputs， timestamps），
            custom_ops={QKVAttention： QKVAttention.count_flops}，
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))

    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def gamma_embedding(gammas, dim, max_period=10000):
    """
  Create a sinusoidal time step embedding.
    :p aram gammas: one-dimensional tensors for N indexes, one per batch element.
                      These may be fractional.
    :p aram dim: The dimension of the output.
    :p ARAM max_period: Controls the minimum frequency of embedding.
    :return: position-embedded [N x dim] tensors.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding