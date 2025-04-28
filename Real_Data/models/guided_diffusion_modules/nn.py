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
    将模块的参数清零并返回.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    缩放模块的参数并返回它。
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    取所有非批次维度的平均值。
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
   创建一个标准的归一化层。

:p aram 通道数：输入通道数。
    ：return： 一个 nn。用于规范化的模块。
    """
    return GroupNorm32(32, channels)



def checkpoint(func, inputs, params, flag):
    """
    在不缓存中间激活的情况下评估函数，从而允许
    减少内存，但代价是向后传递中的额外计算。

:p aram func：要计算的函数。
    :p aram inputs：传递给 'func' 的参数序列。
    :p aram params：参数“func”序列依赖于但不依赖于
                   显式地作为参数。
    :p aram 标志：如果为 False，则禁用梯度检查点。
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
            # 修复了一个 bug，即 run_function 中的第一个运算修改了
            # 张量存储到位，detach（）'d 不允许
            # 张量。
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
    “thop”包的计数器，用于计算
    注意操作。
    旨在像以下方式使用：
        macs，参数 = thop.profile（
            型
            inputs=（inputs， timestamps），
            custom_ops={QKVAttention： QKVAttention.count_flops}，
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
   # 我们用相同数量的运算执行两个 matmuls。
    # 第一个计算权重矩阵，第二个计算
    # 值向量的组合。
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def gamma_embedding(gammas, dim, max_period=10000):
    """
    创建正弦时间步长嵌入。
    :p aram gammas：N 个索引的一维张量，每个批次元素一个。
                      这些可能是分数的。
    :p aram dim：输出的维度。
    :p aram max_period：控制嵌入的最小频率。
    ：return： 位置嵌入的 [N x dim] 张量。
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