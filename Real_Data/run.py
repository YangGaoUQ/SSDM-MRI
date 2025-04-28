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


def main_worker(gpu, ngpus_per_node, opt):

    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu

    if torch.cuda.device_count()>1:
        torch.cuda.set_device(int(opt['local_rank']))
        print('Use gpu {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )

    torch.backends.cudnn.enabled = True
    warnings.warn('torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])


    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create a log file in directory {}\n'.format(opt['path']['experiments_root']))


    phase_loader, val_loader = define_dataloader(phase_logger, opt)  # 如果阶段是测试，val_loader为None。
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]



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

    phase_logger.info('Start model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:

        print("finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/img_restoration.json',
                        help='config JSON ')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    args = parser.parse_args()
    opt = Praser.parse(args)


    ''' DistributedDataParallel(DDP) '''

    if torch.cuda.device_count()>1:
        ngpus_per_node = torch.cuda.device_count()  # 或者 torch.cuda.device_count()\
        print(ngpus_per_node)
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)
