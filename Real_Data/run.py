import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
import torch
import torch.multiprocessing as mp
from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


"""
conda activate pytorch-1.13
cd LQ/MRI_Diffusion

nohup python run.py -p train -c config/img_restoration.json > output.log 2>&1 &
tail -f output.log

ps aux | grep "python run.py -p train -c config/img_restoration.json"
python run.py -p test -c config/img_restoration.json
"""


def main_worker(gpu, ngpus_per_node, opt):
    """ 每个GPU上运行的主要函数 """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu

    if torch.cuda.device_count()>1:
        torch.cuda.set_device(int(opt['local_rank']))
        print('使用GPU {} 进行训练'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''设置随机种子和cuDNN环境'''
    torch.backends.cudnn.enabled = True
    warnings.warn('您选择使用cudnn进行加速。torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    '''设置日志记录器'''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('在目录 {} 中创建日志文件。\n'.format(opt['path']['experiments_root']))

    '''设置网络和数据集'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt)  # 如果阶段是测试，val_loader为None。
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]


    ''' 设置评价指标、损失函数、优化器和调度器 '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    phase_logger.info('开始模型 {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        #phase_writer.close()
        print("finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/img_restoration.json',
                        help='用于配置的JSON文件')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='运行训练或测试', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='每个GPU中的批处理大小')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    ''' 解析配置 '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    #print(torch.cuda.device_count())
    ''' CUDA设备 '''
    # gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    # print(gpu_str)
    # print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))
    # print(opt['distributed'])

    ''' 使用DistributedDataParallel(DDP)和多进程进行多GPU训练 '''
    #[待办事项]: 多台机器上的多GPU
    if torch.cuda.device_count()>1:
        ngpus_per_node = torch.cuda.device_count()  # 或者 torch.cuda.device_count()\
        print(ngpus_per_node)
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)
