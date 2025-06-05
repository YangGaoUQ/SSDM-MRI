import os
import random
from tqdm import tqdm
from strategies import *
from models.network import Network


class InfinityDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, L):
        self.dataset = dataset
        self.L = L

    def __getitem__(self, item):
        idx = random.randint(0, len(self.dataset) - 1)
        r = self.dataset[idx]
        return r

    def __len__(self):
        return self.L

class DiffusionDistillation:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train_student(self, distill_train_loader, teacher_diffusion, student_diffusion, student_lr, device,num_epochs=-1):
        scheduler = self.scheduler
        total_steps = len(distill_train_loader)
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")

        for epoch in range(num_epochs):
            pbar = tqdm(distill_train_loader)
            N = 0
            L_tot = 0
            for data in pbar:
                cond_image = data.get('cond_image').to(device)
                gt_image = data.get('gt_image').to(device)
                mask=data.get('mask').to(device)

                scheduler.zero_grad()
                time = 2 * torch.randint(0, student_diffusion.num_timesteps, (cond_image.shape[0],), device=device)

                loss = teacher_diffusion.distill_loss(student_diffusion, gt_image, cond_image, time, mask=mask)
                L = loss.item()
                L_tot += L
                N += 1
                pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} Loss: {L_tot / N}")
                loss.backward()
                scheduler.step()
                if scheduler.stop(N, total_steps):
                    break

            print(f"Epoch {epoch + 1}/{num_epochs}, Final Loss: {L_tot / N}")
            save_filename = '{}_{}.pth'.format(student_diffusion.num_timesteps, f"steps_Network")
            save_path = os.path.join("checkpoints/distillate", save_filename)
            student_schedule = {
                "train": {
                    "schedule": "cosine",
                    "n_timestep": student_diffusion.num_timesteps,
                }
            }
            student_model = Network(student_diffusion.net_, student_schedule, time_scale=student_diffusion.time_scale, distill=True)
            student_model.set_new_noise_schedule()
            torch.save(student_model.state_dict(), save_path)
