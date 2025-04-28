from models.guided_diffusion_modules.unet import UNet
from data.dataset import MRI_Restoration


def make_model():
    net = UNet(in_channel= 2,
               out_channel= 1,
               inner_channel=64,
               channel_mults= [1,2,4,8],
               attn_res=[8],
               num_head_channels=32,
               res_blocks=2,
               dropout=0.2,
               image_size=320)
    return net

def make_dataset(data_root="train",acc_factor=-1):
        return MRI_Restoration(data_root=data_root,acc_factor=acc_factor)



