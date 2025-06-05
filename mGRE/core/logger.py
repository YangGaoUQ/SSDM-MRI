import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import core.util as Util
from data.dataset import complex_array_to_abs,normalize_image
import nibabel as nib
import matplotlib.pyplot as plt

def plot_tensor(tensor, title="Image"):

    tensor_np = tensor
    tensor_np=np.abs(tensor_np[0,:,:]+1j*tensor_np[1,:,:])

    if np.iscomplexobj(tensor_np):
        tensor_np = np.abs(tensor_np)

    plt.imshow(tensor_np, cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()
class InfoLogger():
    """
    Using logging to keep logs, you can only judge global_rank work on GPU 0
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
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
  Record visual effects with Tensorboard with support for 'add_scalar', 'add_scalars', 'add_image', 'add_images', and more.
    The function of saving results is also integrated.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank==0:
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
        try:
            names = results['name']
            # outputs = Util.postprocess(results['result'])
            outputs=results['result']
            for i in range(len(names)):
                names[i] = os.path.splitext(names[i])[0]
                data_to_save = outputs[i].numpy()
                # if names[i].startswith('Out'):
                #     plot_tensor(data_to_save)
                #     print(data_to_save.shape)
                if not names[i].startswith("mask"):
                    data_to_save_abs, data_to_save_img = complex_array_to_abs(data_to_save)
                    # plot_tensor(data_to_save_img)
                    # real_part = (data_to_save_img.real / np.max(np.abs(data_to_save_img))).astype(np.float32)
                    # imag_part = (data_to_save_img.imag / np.max(np.abs(data_to_save_img))).astype(np.float32)
                    # abs_part = (np.abs(data_to_save_img) / np.max(np.abs(data_to_save_img))).astype(np.float32)
                    np.save(os.path.join(result_path, names[i]),data_to_save_img)
                    real_part = (data_to_save_img.real).astype(np.float32)
                    imag_part = (data_to_save_img.imag).astype(np.float32)
                    phase_part=np.angle(data_to_save_img)
                    #abs_part = (np.abs(data_to_save_img) / np.max(np.abs(data_to_save_img))).astype(np.float32)
                    abs_part = np.abs(data_to_save_img).astype(np.float32)

                    real_nii = nib.Nifti1Image(real_part, np.eye(4))
                    nib.save(real_nii, os.path.join(result_path, names[i] + '_real.nii'))

                    imag_nii = nib.Nifti1Image(imag_part, np.eye(4))
                    nib.save(imag_nii, os.path.join(result_path, names[i] + '_imag.nii'))

                    abs_nii = nib.Nifti1Image(abs_part, np.eye(4))
                    nib.save(abs_nii, os.path.join(result_path, names[i] + '_abs.nii'))

                    phase_nii = nib.Nifti1Image(phase_part, np.eye(4))
                    nib.save(phase_nii, os.path.join(result_path, names[i] + '_phase.nii'))

                    # img222_scaled = (data_to_save_abs * 255).astype(np.uint8)
                    # image_to_save = Image.fromarray(img222_scaled)
                    # image_to_save.save(os.path.join(result_path, names[i] + '.png'))
                else:
                    min_val = np.min(data_to_save)
                    max_val = np.max(data_to_save)

                    # 最大值最小值归一化
                    data_to_save = (data_to_save - min_val) / (max_val - min_val)
                    img222_scaled = (data_to_save * 255).astype(np.uint8)
                    image_to_save = Image.fromarray(img222_scaled)
                    image_to_save.save(os.path.join(result_path, names[i] + '.png'))

        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')



    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
       If the visualization is configured to use:
            Returns the add_data() method of the TensorBoard with additional information (steps, tags).
        Otherwise:
            Returns a blank function handle that does nothing
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

            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    Record the numerical metrics of the training。
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
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
