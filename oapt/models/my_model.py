import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img,get_root_logger
# from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
import oapt.models.utils_image as util
import numpy as np

import math
from tqdm import tqdm
from os import path as osp


def calculate_psnrb(img_E, img_H, border=0, input_order='CHW',test_y_channel=False):
    img_E = util.tensor2single(img_E)
    img_E = util.single2uint(img_E)
    return util.calculate_psnrb(img_H, img_E, border=0)



@MODEL_REGISTRY.register()
class MYModel(SRModel):
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'offset' in data:
            self.offset = data['offset'].to(self.device)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.opt['network_g']['type'].find('Offset')>-1:
            self.output = self.net_g(self.lq, self.offset)
        else:
            self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt) #MyLoss(self.output, self.output_offset, self.gt, self.offset)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        

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
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['network_g']['type'].find('Offset')>-1:
                    self.output = self.net_g_ema(self.img, self.offset)
                else:
                    self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['network_g']['type'].find('Offset')>-1:
                    self.output = self.net_g(self.img, self.offset)
                else:
                    self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
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
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]
    
    def tile_process_8x8(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
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
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]

    def post_process(self):
        if self.opt['network_g']['type'] == 'FBCNN':
            self.output = self.output[0]

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
                if self.offset.any() and self.opt['network_g']['type'].find('Offset')>-1:
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

            

            # psnr_b_record[idx] = calculate_psnrb(self.img,gt_img,border=0,test_y_channel=False)
            # psnr_b_record_out[idx] = calculate_psnrb(self.output,gt_img,border=0,test_y_channel=False)

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
