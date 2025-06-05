import argparse
import importlib
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from v_diffusion import make_beta_schedule
from train_utils import *
from moving_average import init_ema_model
import logging
from datetime import datetime


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():

    dist.destroy_process_group()


def load_UNet_state_dict(path, device):

    checkpoint = torch.load(path, map_location=device)
    unet_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('denoise_fn.'):
            unet_state_dict[key[11:]] = value
    return unet_state_dict


def make_diffusion(model, n_timestep, time_scale, device):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
    M = importlib.import_module("v_diffusion")
    D = getattr(M, args.diffusion)
    r = D(model, betas, time_scale=time_scale)
    r.gamma = args.gamma
    return r

def make_scheduler():
    M = importlib.import_module("train_utils")
    D = getattr(M, args.scheduler)
    return D()


def load_teacher_model(path, n_timesteps, time_scale, device):
    ckpt = load_UNet_state_dict(path, device)
    teacher_ema = make_model().to(device)
    teacher_ema.load_state_dict(ckpt)


    if torch.cuda.device_count() > 1 and not args.disable_ddp:
        teacher_ema = DDP(teacher_ema, device_ids=[device])

    teacher_ema_diffusion = make_diffusion(teacher_ema, n_timesteps, time_scale, device)
    return teacher_ema, teacher_ema_diffusion

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/200_Network.pth")
    parser.add_argument("--data_root", type=str, default="train")
    parser.add_argument("--n_timesteps", type=int, default=2)
    parser.add_argument("--time_scale", type=int, default=1)
    parser.add_argument("--target_steps", type=int, default=1)
    parser.add_argument("--acc_factor", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--module", type=str, default="celeba_u")
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--num_iters", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.3 * 5e-5)
    parser.add_argument("--scheduler", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", type=int, default=1000000)
    parser.add_argument("--ckpt_interval", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def train_worker(rank, world_size, args, make_model, make_dataset):
    if world_size > 1:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        args.batch_size = args.batch_size // world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42 + rank)

    train_dataset = InfinityDataset(
        make_dataset(data_root=args.data_root, acc_factor=args.acc_factor),
        args.num_iters
    )


    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    ) if world_size > 1 else None

    distill_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    scheduler = make_scheduler()
    distillation_model = DiffusionDistillation(scheduler)

    teacher_ema, teacher_ema_diffusion = load_teacher_model(
        args.base_checkpoint,
        args.n_timesteps,
        args.time_scale,
        device
    )

    current_steps = teacher_ema_diffusion.num_timesteps
    while current_steps > args.target_steps:
        student = make_model().to(device)
        init_ema_model(teacher_ema, student, device)
        if world_size > 1 and not args.disable_ddp:
            student = DDP(student, device_ids=[rank])

        student_diffusion = make_diffusion(
            student,
            teacher_ema_diffusion.num_timesteps // 2,
            teacher_ema_diffusion.time_scale * 2,
            device
        )
        distillation_model.train_student(
            distill_train_loader,
            teacher_ema_diffusion,
            student_diffusion,
            args.lr,
            device,
            num_epochs=args.epoch
        )

        save_filename = f"{student_diffusion.num_timesteps}_steps_Network.pth"
        save_path = os.path.join("checkpoints/distillate", save_filename)

        teacher_ema, teacher_ema_diffusion = load_teacher_model(
            save_path if rank == 0 else args.base_checkpoint,
            student_diffusion.num_timesteps,
            student_diffusion.time_scale,
            device
        )
        current_steps = student_diffusion.num_timesteps

    if world_size > 1:
        cleanup()

def distill_model(args, make_model, make_dataset):
    world_size = torch.cuda.device_count() if not args.disable_ddp and torch.cuda.device_count() > 1 else 1
    if world_size > 1:
        print(f"Using {world_size} GPUs with DistributedDataParallel")
        mp.spawn(
            train_worker,
            args=(world_size, args, make_model, make_dataset),
            nprocs=world_size,
            join=True
        )
    else:
        print("Using single GPU or CPU")
        train_worker(0, 1, args, make_model, make_dataset)


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    distill_model(args, make_model, make_dataset)