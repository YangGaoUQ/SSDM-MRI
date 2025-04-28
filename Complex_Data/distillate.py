# coding: utf-8
import argparse
import importlib
from v_diffusion import make_beta_schedule
from train_utils import *
from moving_average import init_ema_model
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings


def make_argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", help="Path to base checkpoint.", type=str,
                        default="checkpointh")
    parser.add_argument("--n_timesteps",default=16)
    parser.add_argument("--time_scale", default=1)
    parser.add_argument("--target_steps", help="target distillate steps", default=1)
    parser.add_argument("--AF", help="undersampling factor", default=4)
    parser.add_argument("--epoch",default=2)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=20000)

    parser.add_argument("--module", help="Model module.", type=str, default="celeba_u")
    parser.add_argument("--gamma", help="Gamma factor for SNR weights.", type=float, default=0)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=2)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.3 * 5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=1000000)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser

def distill_model(gpu,args, make_model, make_dataset):
    def load_UNet_state_dict(path):
        checkpoint = torch.load(path)
        # Extract the parameters of the UNet section from the pretrained model
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
        ckpt = load_UNet_state_dict(path)
        teacher_ema = make_model().to(device)
        teacher_ema.load_state_dict(ckpt)
        teacher_ema_diffusion = make_diffusion(teacher_ema, n_timesteps, time_scale, device)

        return teacher_ema, teacher_ema_diffusion
    # Sets the number of processes on the loaded dataset
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    #Distributed training
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(int(gpu))
        print('Use GPU {} for training'.format(int(gpu)))
        dist.init_process_group(backend='nccl',
                                init_method="tcp://127.0.0.1:21012",
                                world_size=2,
                                rank=gpu,
                                group_name='mtorch'
                                )

    torch.backends.cudnn.enabled = True
    warnings.warn('You choose to use cudnn for acceleration。torch.backends.cudnn.enabled=True')
    device = torch.device("cuda")

    #Load the dataset
    train_dataset = InfinityDataset(make_dataset(AF=args.AF), args.num_iters)
    distill_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    #create distillation model
    scheduler = make_scheduler()
    distillation_model = DiffusionDistillation(scheduler)

    #Teacher model UNet checkpoint
    teacher_ema,teacher_ema_diffusion=load_teacher_model(args.base_checkpoint,args.n_timesteps,args.time_scale,device)
    print(f"Num timesteps: {teacher_ema_diffusion.num_timesteps}, time scale: {teacher_ema_diffusion.time_scale}.")

    current_steps=teacher_ema_diffusion.num_timesteps
    #If the refining step does not reach the target step, the student model will be positioned as a new teacher model and the training will continue
    while current_steps>args.target_steps:
        #Create a student model and initialize it with a teacher model
        student = make_model().to(device)
        init_ema_model(teacher_ema, student, device)
        print("Teacher parameters copied.")

        #Create a student model
        student_diffusion = make_diffusion(student, teacher_ema_diffusion.num_timesteps // 2,
                                           teacher_ema_diffusion.time_scale * 2, device)

        #Train student models
        distillation_model.train_student(distill_train_loader, teacher_ema_diffusion, student_diffusion,
                                         args.lr, device,num_epochs=args.epoch)
        print(f"Distillate {student_diffusion.num_timesteps} steps model finish")

        save_filename = '{}_{}.pth'.format(student_diffusion.num_timesteps, "steps_Network")
        save_path = os.path.join("checkpoint/distillate", save_filename)
        teacher_ema, teacher_ema_diffusion = load_teacher_model(save_path, student_diffusion.num_timesteps, student_diffusion.time_scale,device)
        current_steps=student_diffusion.num_timesteps
    print("All Finished.")

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if torch.cuda.device_count()>1:
        ngpus_per_node = torch.cuda.device_count()
        print(ngpus_per_node)
        mp.spawn(distill_model, nprocs=ngpus_per_node, args=(args, make_model, make_dataset))
    else :
        distill_model(0,args, make_model, make_dataset)
