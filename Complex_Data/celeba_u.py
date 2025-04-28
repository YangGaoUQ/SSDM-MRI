from train_utils import *
from models.guided_diffusion_modules.unet import UNet
from data.dataset import MyDataSet

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1

def make_model():
    net = UNet(in_channel= 4,
               out_channel= 2,
               inner_channel=64,
               channel_mults= [1,2,4,8],
               attn_res=[16],
               num_head_channels=32,
               res_blocks=2,
               dropout=0.2,
               image_size=256)
    return net

def make_dataset(phase="train",AF=-1):
    if phase=="train":
        return MyDataSet(data_root="imgs",AF=AF)
    else:
        return MyDataSet(data_root="test_dataset",AF=AF)


