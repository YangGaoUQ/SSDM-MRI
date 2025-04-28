import os
import random
from tqdm import tqdm
from strategies import *
from models.network import Network
from discriminator import SimplePatchGANDiscriminator
from losses import discriminator_loss, generator_loss
import torch.nn.functional as F

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

        discriminator = SimplePatchGANDiscriminator(input_channels=1).float().to(device)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # 超参数设置
        lambda_adv = 0.001  # 对抗损失权重
        lambda_mse = 1.0  # 原始MSE损失权重
        d_update_freq = 2  # 判别器更新频率（每2步更新一次）

        for epoch in range(num_epochs):
            L_tot = 0
            N = 0
            pbar = tqdm(distill_train_loader)

            for step, data in enumerate(pbar):
                # 数据加载
                cond_image = data.get('cond_image').to(device)
                gt_image = data.get('gt_image').to(device)
                mask = data.get('mask').to(device)

                time = 2 * torch.randint(0, student_diffusion.num_timesteps,(cond_image.shape[0],), device=device)
                v, v_t = teacher_diffusion.distill_loss(student_diffusion, gt_image, cond_image, time,mask=mask)
                mse_loss = F.mse_loss(v.float(), v_t.float())

                if step % d_update_freq == 0:
                    optimizer_D.zero_grad()
                    d_loss = discriminator_loss(discriminator, v.detach(), v_t.detach())
                    d_loss.backward()
                    optimizer_D.step()

                scheduler.zero_grad()
                adv_loss = generator_loss(discriminator, v.float())  # 确保 v 是 float32
                total_loss = lambda_mse * mse_loss + lambda_adv * adv_loss
                total_loss = total_loss.float()  # 强制转换为 float32

                total_loss.backward()
                scheduler.step()

                # 记录损失
                L = total_loss.item()
                L_tot += L
                N += 1
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Loss: {L_tot / N:.6f} (MSE: {mse_loss.item():.6f}, "
                    f"Adv: {adv_loss.item():.6f})"
                )

                if scheduler.stop(N, total_steps):
                    break

        # for epoch in range(num_epochs):
        #     pbar = tqdm(distill_train_loader)
        #     N = 0
        #     L_tot = 0
        #     for data in pbar:
        #         #欠采样图像和 原图像
        #         cond_image = data.get('cond_image').to(device)
        #         gt_image = data.get('gt_image').to(device)
        #         mask=data.get('mask').to(device)
        #
        #         scheduler.zero_grad()
        #         time = 2 * torch.randint(0, student_diffusion.num_timesteps, (cond_image.shape[0],), device=device)
        #         loss = teacher_diffusion.distill_loss(student_diffusion, gt_image, cond_image, time,mask=mask)
        #         L = loss.item()
        #         L_tot += L
        #         N += 1
        #         pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {L_tot / N}")
        #         loss.backward()
        #         scheduler.step()
        #         if scheduler.stop(N, total_steps):
        #             break

            print(f"Epoch {epoch + 1}/{num_epochs}, Final Loss: {L_tot / N}")

            #学生模型保存
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
