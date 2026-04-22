from functools import partial
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset
import core.util as Util
from core.praser import init_obj

def define_dataloader(logger, opt):
    """ 创建训练/测试数据加载器和验证数据加载器。在测试阶段或非 GPU 0 上时，验证数据加载器为None。"""
    '''创建数据集并设置随机种子'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''创建数据采样器'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False),
                                          num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle': False})  # 采样器选项与 shuffle 互斥

    '''创建数据加载器和验证数据加载器'''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    '''验证数据加载器不使用 DistributedSampler，只在 GPU 0 上运行！'''
    if opt['global_rank'] == 0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args', {}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args)
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    '''从给定的文件名加载 Dataset() 类'''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    print(logger)
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='dataset.dataset', init_type='Dataset')
    val_dataset = None

    valid_len = 0
    data_len = len(phase_dataset)


    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = dataloder_opt.get('validation_split', 0)

    '''划分验证数据集，当阶段为测试或 validation_split 为 0 时，valid_split==0。'''
    if valid_split > 0.0 or 'debug' in opt['name']:
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        data_len -= valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len],
                                                  generator=Generator().manual_seed(opt['seed']))

    logger.info('{} 阶段的数据集有 {} 个样本。'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        logger.info('{} 阶段的数据集有 {} 个样本。'.format('val', valid_len))
    return phase_dataset, val_dataset


def subset_split(dataset, lengths, generator):
    """
    将数据集分割成给定长度的非重叠新数据集。主要代码来自 pytorch 中的 random_split 函数。
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length: offset]))
    return Subsets
