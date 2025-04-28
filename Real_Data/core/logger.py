import os

import numpy as np
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import core.util as Util
import cv2
import nibabel as nib


class InfoLogger():
    """
    使用日志记录来记录日志，只能通过判断global_rank在GPU 0上工作
    """

    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0:  # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass

            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)

            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)

            return wrapper

    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)


class VisualWriter():
    """
   使用 Tensorboard 录制视觉效果，支持 'add_scalar'、'add_scalars'、'add_image'、'add_images'等功能。
    还集成了保存结果功能。
    """

    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank == 0:
            log_dir = str(log_dir)

            # 检索 vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                          "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' 从结果中获取名称和相应的图像[OrderedDict] '''
        try:
            names = results['name']
            # save nii npy
            outputs = Util.postprocess_nii_npy(results['result'])
            for i in range(len(names)):
                names[i] = os.path.splitext(names[i])[0]
                if not names[i].startswith("Process"):
                    nifti_image = nib.Nifti1Image(outputs[i], np.eye(4))  # 假设输出是一个numpy数组，并使用单位矩阵作为仿射变换
                    nifti_file_name = names[i] + '.nii'
                    nib.save(nifti_image, os.path.join(result_path, nifti_file_name))
                    np.save(os.path.join(result_path, names[i]), outputs[i])
        except:
            raise NotImplementedError(
                'You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

    def __getattr__(self, name):
        """
        如果可视化配置为使用：
            返回 TensorBoard 的 add_data（） 方法，并添加了附加信息（步骤、标记）。
        否则：
            返回一个不执行任何操作的空白函数句柄
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)

            return wrapper
        else:
            # 返回此类中定义的方法的默认操作，例如 set_step（）。
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    记录训练数值指标。
    """

    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k): v for k, v in dict(self._data.average).items()}
