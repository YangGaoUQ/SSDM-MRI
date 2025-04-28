import argparse
import importlib
import torch
import os
from v_diffusion import make_beta_schedule
from train_utils import *
from moving_average import init_ema_model


def load_UNet_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    unet_state_dict = {k[11:]: v for k, v in checkpoint.items() if k.startswith('denoise_fn.')}
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
    ckpt = load_UNet_state_dict(path)
    teacher_ema = make_model().to(device)
    teacher_ema.load_state_dict(ckpt)
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
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--module", type=str, default="celeba_u")
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--num_iters", type=int, default=55694)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.3 * 5e-5)
    parser.add_argument("--scheduler", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def distill_model(args, make_model, make_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs" if num_gpus > 1 else "Using single GPU")

    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    train_dataset = InfinityDataset(make_dataset(data_root=args.data_root, acc_factor=args.acc_factor), args.num_iters)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    scheduler = make_scheduler()
    distillation_model = DiffusionDistillation(scheduler)

    teacher_ema, teacher_ema_diffusion = load_teacher_model(args.base_checkpoint, args.n_timesteps, args.time_scale,
                                                            device)
    print(f"Num timesteps: {teacher_ema_diffusion.num_timesteps}, time scale: {teacher_ema_diffusion.time_scale}.")

    current_steps = teacher_ema_diffusion.num_timesteps
    while current_steps > args.target_steps:
        student = make_model().to(device)
        init_ema_model(teacher_ema, student, device)
        print("Teacher parameters copied.")

        if num_gpus > 1:
            student = torch.nn.DataParallel(student)

        student_diffusion = make_diffusion(student, current_steps // 2, teacher_ema_diffusion.time_scale * 2, device)

        log_losses = []
        for epoch in range(args.epoch):
            epoch_loss = distillation_model.train_student(train_loader, teacher_ema_diffusion, student_diffusion,
                                                          args.lr, device, num_epochs=args.epoch)
            log_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")

        with open("training_log.txt", "a") as log_file:
            log_file.write(f"Steps {student_diffusion.num_timesteps}: {log_losses}\n")

        save_filename = f"{student_diffusion.num_timesteps}_steps_Network.pth"
        save_path = os.path.join("checkpoints/distillate", save_filename)
        torch.save(student.state_dict(), save_path)

        teacher_ema, teacher_ema_diffusion = load_teacher_model(save_path, student_diffusion.num_timesteps,
                                                                student_diffusion.time_scale, device)

    print("All Finished.")


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")
    distill_model(args, make_model, make_dataset)
