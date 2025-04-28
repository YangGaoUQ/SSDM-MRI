import os
from collections import OrderedDict
import json
from pathlib import Path
from datetime import datetime
from functools import partial
import importlib
from types  import FunctionType
import shutil
import  torch
def init_obj(opt, logger, *args, default_file_name='default file', given_module=None, init_type='Network', **modify_kwargs):
    """
    在配置中查找名称为“name”的函数句柄，
    并返回使用相应参数初始化的实例。
    """
    if opt is None or len(opt)<1:
        logger.info('Option is None when initialize {}'.format(init_type))
        return None
    
    ''' 默认格式为带名称键的 dict '''
    if isinstance(opt, str):
        opt = {'name': opt}
        logger.warning('Config is a str, converts to a dict {}'.format(opt))

    name = opt['name']
    ''' name 可以是 list，表示函数的文件和类名 '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name

    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)

        attr = getattr(module, class_name)
        kwargs = opt.get('args', {})
        kwargs.update(modify_kwargs)

        ''' 使用 args 导入类或函数 '''
        if isinstance(attr, type):
            ret = attr(*args, **kwargs)
            ret.__name__  = ret.__class__.__name__


        elif isinstance(attr, FunctionType):
            ret = partial(attr, *args, **kwargs)
            ret.__name__  = attr.__name__
            # ret = attr
        logger.info('{} [{:s}() form {:s}] is created.'.format(init_type, class_name, file_name))
    except:
        raise NotImplementedError('{} [{:s}() from {:s}] not recognized.'.format(init_type, class_name, file_name))
    return ret


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict 到 logger 的字符串 """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse(args):
    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    ''' 使用 args 替换配置上下文 '''
    opt['phase'] = args.phase
    if args.gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in args.gpu_ids.split(',')]
    if args.batch is not None:
        opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = args.batch
 
    ''' 设置 CUDA 环境 '''
    if torch.cuda.device_count() > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    ''' 更新名称 '''
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    elif opt['finetune_norm']:
        opt['name'] = 'finetune_{}'.format(opt['name'])
    else:
        opt['name'] = '{}_{}'.format(opt['phase'], opt['name'])

    ''' 设置日志目录 '''
    experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}'.format(opt['name'], get_timestamp()))
    mkdirs(experiments_root)

    ''' 保存 JSON'''
    write_json(opt, '{}/config.json'.format(experiments_root))

    ''' 更改文件夹相对层次结构 '''
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    ''' 调试模式 '''
    if 'debug' in opt['name']:
        opt['train'].update(opt['debug'])

    ''' 代码备份 '''
    for name in os.listdir('.'):
        if name in ['config', 'models', 'core', 'slurm', 'dataset']:
            shutil.copytree(name, os.path.join(opt['path']['code'], name), ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
        if '.py' in name or '.sh' in name:
            shutil.copy(name, opt['path']['code'])
    return dict_to_nonedict(opt)





