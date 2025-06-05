from train_utils import *
from models.guided_diffusion_modules.unet import UNet
from data.dataset import MyDataSet


def make_model():
    net = UNet(in_channel= 4,
               out_channel= 2,
               inner_channel=64,
               channel_mults= [1,2,4,8],
               attn_res=[8],
               num_head_channels=32,
               res_blocks=2,
               dropout=0.2,
               image_size=256)
    return net

def make_dataset(data_root="train",acc_factor=-1):
        return MyDataSet(data_root=data_root,acc_factor=acc_factor)



