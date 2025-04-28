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
import matplotlib.pyplot as plt


def main_worker(gpu, ngpus_per_node, opt):
    """ The main functions running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu

    if torch.cuda.device_count()>1:
        torch.cuda.set_device(int(opt['local_rank']))
        print('Use GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''Set up a random seed and cu DNN environment'''
    torch.backends.cudnn.enabled = True
    warnings.warn('You choose to use cudnn for acceleration.torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    '''Set up a logger'''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('在目录 {} 中创建日志文件。\n'.format(opt['path']['experiments_root']))

    '''Set up networks and datasets'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt)  # 如果阶段是测试，val_loader为None。
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]


    ''' Set up evaluation metrics, loss functions, optimizers, and schedulers '''
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

    phase_logger.info('Start the model {}.'.format(opt['phase']))
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
                        help='A JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run a training or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='The size of the batch in each GPU')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    ''' Parse configuration '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    #print(torch.cuda.device_count())
    ''' CUDA device '''
    # gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    # print(gpu_str)
    # print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))
    # print(opt['distributed'])

    ''' use DistributedDataParallel(DDP)and multi-process multi-GPU training '''

    if torch.cuda.device_count()>1:
        ngpus_per_node = torch.cuda.device_count()  # 或者 torch.cuda.device_count()\
        print(ngpus_per_node)
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)
