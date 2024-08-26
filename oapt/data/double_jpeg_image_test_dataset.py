from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import os
import os.path as osp
import numpy as np
import random
import torch


@DATASET_REGISTRY.register()
class DoubleJpegImageDataset_test(data.Dataset): # only for unkown testing
    """ image dataset for jpeg image restoration (only lq image).
    """
    def __init__(self, opt):
        super(DoubleJpegImageDataset_test, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.num_channels = opt['num_channels']
            
        # no use
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # dataset
        if 'multi_dataroot' in opt:
            self.paths = []
            logger = get_root_logger()
            for k, v in opt['multi_dataroot'].items():
                gt_folder, lq_folder = v['dataroot_gt'], v['dataroot_lq']
                self.paths += paired_paths_from_folder_recursive0([lq_folder, gt_folder], ['lq', 'gt'],
                                                                self.filename_tmpl, opt['phase'])
                # logger.info(f'sample of dataset : {self.paths[0]}')
                logger.info(f'sample of dataset : {self.paths[-1]}')
        else:
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            if self.io_backend_opt['type'] == 'lmdb':
                print("lmdb not inpletment")
            elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                print("meta_info not inpletment")
            else:
                self.paths = paired_paths_from_folder_recursive0([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                                self.filename_tmpl, opt['phase'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        if self.opt['name'].find('Classic5')>-1:
            flag = 'grayscale'
        else:
            flag = 'color'
        img_lq = imfrombytes(img_bytes, flag, float32=False)
        img_gt = imfrombytes(img_bytes, flag, float32=False)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y' and flag == 'color':
            img_gt = bgr2ycbcr0(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr0(img_lq, y_only=True)[..., None]

        
        if flag == 'grayscale':
            img_gt = np.expand_dims(img_gt,-1)
            img_lq = np.expand_dims(img_lq,-1)

        img_lq = img_lq.astype(np.float32) / 255.
        img_gt = img_gt.astype(np.float32) / 255.
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
         
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'lq_path': lq_path, 'gt': img_gt, 'gt_path': lq_path}

    def __len__(self):
        return len(self.paths)



def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def _convert_input_type_range0(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range0(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr0(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range0(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range0(out_img, img_type)
    return out_img


def paired_paths_from_folder_recursive0(folders, keys, filename_tmpl, phase):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                            f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = sorted(list(scandir(input_folder, recursive=True, full_path=True)))
    gt_paths = sorted(list(scandir(gt_folder, recursive=True, full_path=True)))
    assert len(input_paths) == len(gt_paths), (f'{input_folder} and {gt_folder} datasets have different number of images: '
                                            f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []

    for input_path, gt_path in zip(input_paths, gt_paths):
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))

    return paths

def paired_random_crop0(img_gts, img_lqs, gt_patch_size, scale, gt_path=None, chop_8x8=False):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size list (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = [gt_patch_size[0] // scale, gt_patch_size[1] // scale]

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size[0] or w_lq < lq_patch_size[1]:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size[0]}, {lq_patch_size[1]}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    if chop_8x8:
        top = random.randint(0, (h_lq - lq_patch_size[0]) // 8 - 1) * 8
        left = random.randint(0, (w_lq - lq_patch_size[1]) // 8 - 1) * 8
    else:
        top = random.randint(0, h_lq - lq_patch_size[0])
        left = random.randint(0, w_lq - lq_patch_size[1])


    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size[0], left:left + lq_patch_size[1]] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size[0], left:left + lq_patch_size[1], ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[0]] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1], ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs