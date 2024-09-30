import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img,get_root_logger #imwrite是cv2的
# from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
import oapt.models.utils_image as util
import numpy as np
from oapt.matlab_functions import rgb2ycbcr, ycbcr2rgb, bgr2ycbcr, ycbcr2bgr
import cv2

import math
from tqdm import tqdm
from os import path as osp

def rgb2ycbcr1(tensor_x):
    y = tensor_x.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    y = rgb2ycbcr(y).transpose(2,0,1)
    y = torch.tensor(y).unsqueeze(0).to(tensor_x.device)
    return y

def ycbcr2rgb1(tensor_x):
    y = tensor_x.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    y = ycbcr2rgb(y).transpose(2,0,1)
    y = torch.tensor(y).unsqueeze(0).to(tensor_x.device)
    return y

def bgr2ycbcr1(tensor_x):
    y = tensor_x.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    y = bgr2ycbcr(y).transpose(2,0,1)
    y = torch.tensor(y).unsqueeze(0).to(tensor_x.device)
    return y

def ycbcr2bgr1(tensor_x):
    y = tensor_x.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    y = ycbcr2bgr(y).transpose(2,0,1)
    y = torch.tensor(y).unsqueeze(0).to(tensor_x.device)
    return y

def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def _calculate_psnrb(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR-B (Peak Signal-to-Noise Ratio).

    Ref: Quality assessment of deblocked images, for JPEG image deblocking evaluation
    # https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py

    Args:
        img1 (ndarray): Images with range [0, 255]. lq
        img2 (ndarray): Images with range [0, 255]. gt
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]

def calculate_psnrb1(img1, img2, border=0, input_mode='output', test_y_channel=False):
    if input_mode == 'output':
        img1 = img1.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img1 = (img1 * 255.0).round().astype(np.uint8)  # float32 to uint8
        if len(img1.shape)==3:
            img1 = img1.transpose(1,2,0)
    psnrb = _calculate_psnrb(img1, img2, crop_border=border, test_y_channel=test_y_channel)
    return psnrb


def calculate_psnrb(img_E, img_H, border=0, input_mode='output',test_y_channel=True):
    if input_mode=='output':
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)
    return util.calculate_psnrb(img_H, img_E, border=0)

@MODEL_REGISTRY.register()
class MYModelY(SRModel):
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'offset' in data:
            self.offset = data['offset'].to(self.device)

    def pre_process(self):
        # pad to multiplication of window_size
        if 'window_size' in self.opt['network_g']:
            window_size = self.opt['network_g']['window_size']
            self.scale = self.opt.get('scale', 1)
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if isinstance(window_size, int):
                window_size = window_size
            else:
                window_size = window_size[0]
            if h % window_size != 0:
                self.mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                self.mod_pad_w = window_size - w % window_size
            self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
        else:
            self.img = self.lq
            self.scale = self.opt.get('scale', 1)
            self.mod_pad_h = 0
            self.mod_pad_w = 0

    def process(self):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        self.output = self.img.new_zeros(output_shape)
        process_y = False
        if self.img.shape[1] == 3:
            print("only test on the Y channel !")
            process_y = True
            self.img = rgb2ycbcr1(self.img)
            self.output[:,1:,:,:] = self.img[:,1:,:,:]
            self.img = self.img[:,0,:,:].unsqueeze(0)
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['network_g']['type'].find('Offset')>-1:
                    self.output[:,0,:,:] = self.net_g_ema(self.img, self.offset)
                elif self.opt['network_g']['type'] == 'FBCNN':
                    self.output[:,0,:,:]  = self.net_g_ema(self.img)[0]
                else:
                    self.output[:,0,:,:]  = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['network_g']['type'].find('Offset')>-1:
                    self.output[:,0,:,:]  = self.net_g(self.img, self.offset)
                elif self.opt['network_g']['type'] == 'FBCNN':
                    self.output[:,0,:,:]  = self.net_g(self.img)[0]
                else:
                    self.output[:,0,:,:]  = self.net_g(self.img)
            # self.net_g.train()
        if process_y:
            self.output = ycbcr2rgb1(self.output)
        # self.output = ycbcr2rgb1(self.output)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        process_y = False

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        if self.img.shape[1] == 3:
            print("only test on the Y channel !")
            process_y = True
            self.img = rgb2ycbcr1(self.img)
            self.output[:,1:,:,:] = self.img[:,1:,:,:]
            self.img = self.img[:,0,:,:].unsqueeze(0)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # print(input_tile.shape)

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            if self.opt['network_g']['type'].find('Offset')>-1:
                                output_tile = self.net_g_ema(input_tile, self.offset)
                            else:
                                output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            if self.opt['network_g']['type'].find('Offset')>-1:
                                output_tile = self.net_g(input_tile, self.offset)
                            else:
                                output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                    print(input_tile.shape)
                # print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, 0, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]
        if process_y:
            self.output = ycbcr2rgb1(self.output)
            # out_img = self.output.squeeze(0).float().detach().cpu().clamp_(0,1).numpy().transpose(1,2,0)
            # out_img = (out_img * 255.).astype(np.uint8)
            # cv2.imwrite("A.png",out_img)
    
    def tile_process_8x8(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        process_y = False

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        if self.img.shape[1] == 3:
            print("only test on the Y channel !")
            process_y = True
            self.img = rgb2ycbcr1(self.img)
            self.output[:,1:,:,:] = self.img[:,1:,:,:]
            self.img = self.img[:,0,:,:].unsqueeze(0)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                # input_start_x_pad必须是8的倍数
                if input_start_x_pad % 8 != 0 and input_start_x_pad % 8 != 8 :
                    input_start_x_pad = max(input_start_x_pad - input_start_x_pad % 8, 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)

                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                # input_start_y_pad必须是8的倍数
                if input_start_y_pad % 8 != 0 and input_start_y_pad % 8 != 8 :
                    input_start_y_pad = max(input_start_y_pad - input_start_y_pad % 8, 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # print(input_tile.shape)

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            if self.opt['network_g']['type'].find('Offset')>-1:
                                output_tile = self.net_g_ema(input_tile, self.offset)
                            else:
                                output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            if self.opt['network_g']['type'].find('Offset')>-1:
                                output_tile = self.net_g(input_tile, self.offset)
                            else:
                                output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                    print(input_tile.shape)
                # print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, 0, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]
        if process_y:
            self.output = ycbcr2rgb1(self.output)
            # out_img = self.output.squeeze(0).float().detach().cpu().clamp_(0,1).numpy().transpose(1,2,0)
            # out_img = cv2.cvtColor((out_img * 255.).astype(np.uint8),cv2.COLOR_RGB2BGR)
            # cv2.imwrite("A.png",out_img)

    def post_process(self):
        # if self.opt['network_g']['type'] == 'FBCNN':
        #     self.output = self.output[0]

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        psnr_b_record_out = np.zeros(len(dataloader)).astype('float')
        psnr_b_record = np.zeros(len(dataloader)).astype('float')
        psnr_record = np.zeros(len(dataloader)).astype('float')
        ssim_record = np.zeros(len(dataloader)).astype('float')
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                if self.opt['network_g']['type'].find('Offset')>-1:
                    self.tile_process_8x8()
                else:
                    self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            if 'lq' in visuals:
                lq_img = tensor2img([visuals['lq']])
                metric_data['img3'] = lq_img
                del self.lq

            psnr_record[idx] = calculate_psnr(lq_img,gt_img,crop_border=0,test_y_channel=False)
            ssim_record[idx] = calculate_ssim(lq_img,gt_img,crop_border=0,test_y_channel=False)

            psnr_b_record[idx] = calculate_psnrb1(lq_img,gt_img,input_mode='input')
            psnr_b_record_out[idx] = calculate_psnrb1(self.output,gt_img,input_mode='output')

            # tentative for out of GPU memory
            
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name, f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_img_path.split(f'_{current_iter}')[0]+'_gt.png')
                imwrite(lq_img, save_img_path.split(f'_{current_iter}')[0]+f'_lq.png')

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    print(calculate_metric(metric_data, opt_))
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            logger = get_root_logger()
            logger.info('out_img: psnrb={:.4f}'.format(np.mean(psnr_b_record_out)))
            logger.info('lq_img: psnr={:.4f};  ssim={:.4f};   psnrb={:.4f}\n'.format(np.mean(psnr_record),np.mean(ssim_record),np.mean(psnr_b_record)))
