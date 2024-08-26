import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img,get_root_logger
from basicsr.archs import build_network
from basicsr.metrics.psnr_ssim import calculate_psnr,calculate_ssim
from oapt.models.my_model import calculate_psnrb
import numpy as np
import math
from tqdm import tqdm
from os import path as osp
from basicsr.losses import build_loss
from copy import deepcopy



@MODEL_REGISTRY.register()
class OAPT_Model(BaseModel): 
    def __init__(self, opt):
        super(OAPT_Model, self).__init__(opt) 
        # define network
        self.net_g = build_network(opt['network_g'])
        # freeze before sending to device
        self.predictor_not_freeze = opt['network_g'].get('predictor_not_freeze', True)
        if not self.predictor_not_freeze: #freeze predictor
            for k, v in self.get_bare_model(self.net_g).prediction.named_parameters():
                v.requires_grad = False
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_pred = self.opt['path'].get('pretrain_network_g_pred', None)
        load_path_test = self.opt['path'].get('pretrain_network_g_rest', None)
        load_path = [load_path, load_path_pred, load_path_test]
        if not all(i is None for i in load_path):
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_networks(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
            logger = get_root_logger()
            net = self.get_bare_model(self.net_g)
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            logger.info(f'Network trainable paramters: {trainable_num:,d}')
        
        self.print_network(self.net_g)
        
    
    def load_networks(self, net, load_path, strict=True, param_key='params'):
        if load_path[0] is not None:
            self.load_network(net, load_path[0], strict, param_key)
        else:
            if self.opt['num_gpu']>0:
                net = self.get_bare_model(net)
            if load_path[1] is not None:
                self.load_network(net.prediction, load_path[1], strict, param_key)
            if load_path[2] is not None:
                self.load_network(net.restoration, load_path[2], strict, param_key)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.
        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if 'params' not in load_net and 'params_ema' not in load_net:
                load_net = load_net
                param_key = 'non-key'
            else:
                if param_key not in load_net and 'params' in load_net:
                    param_key = 'params'
                    logger.info('Loading: params_ema does not exist, use params.')
                load_net = load_net[param_key]
            logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            # load_path = self.opt['path'].get('pretrain_network_g', None)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            load_path_pred = self.opt['path'].get('pretrain_network_g_pred', None)
            load_path_test = self.opt['path'].get('pretrain_network_g_rest', None)
            load_path = [load_path, load_path_pred, load_path_test]
            if not all(i is None for i in load_path): #load_path is not None:
                self.load_networks(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('offset_opt'):
            self.cri_offset = build_loss(train_opt['offset_opt']).to(self.device)
        else:
            self.cri_offset = None

        if self.cri_pix is None and self.cri_offset is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_restore_params = []
        for k, v in self.get_bare_model(self.net_g).restoration.named_parameters():
            if v.requires_grad:
                optim_restore_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} from Restorator will not be optimized.')
        
        optim_pred_params = []
        for k, v in self.get_bare_model(self.net_g).prediction.named_parameters():
            if v.requires_grad:
                optim_pred_params.append(v)
            else:
                # v.requires_grad = False
                logger = get_root_logger()
                logger.warning(f'Params {k} from Predictor will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_restore_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        if self.predictor_not_freeze:
            optim_pred_type = train_opt['optim_g_pred'].pop('type')
            self.optimizer_g_pred = self.get_optimizer(optim_pred_type, optim_pred_params, **train_opt['optim_g_pred'])
            self.optimizers.append(self.optimizer_g_pred)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'offset' in data:
            self.offset = data['offset'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.predictor_not_freeze: #
            self.optimizer_g_pred.zero_grad()
        if self.opt['network_g']['type'].find('SwinIROffsetDenseShift_hw_pred')>-1:
            self.offset_pred, self.output = self.net_g(self.lq)
        else:
            print("only for SwinIROffsetDenseShift_hw_pred")
            sys.exit(0)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt) #MyLoss(self.output, self.output_offset, self.gt, self.offset)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_offset:# and self.predictor_not_freeze:
            l_offset = self.cri_offset(self.offset_pred, self.offset/1.)
            l_total += l_offset
            loss_dict['l_offset'] = l_offset
        l_total.backward()
        self.optimizer_g.step()
        if self.predictor_not_freeze: #
            self.optimizer_g_pred.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

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
                            if self.opt['network_g']['type'].find('SwinIROffsetDenseShift_hw_pred')>-1:
                                if y==0 and x==0:
                                    offset_pred, output_tile = self.net_g_ema.test_forward(input_tile)
                                else:
                                    _, output_tile = self.net_g_ema.test_forward(input_tile, offset_pred, test=True)

                            else:
                                print("only for SwinIROffsetDenseShift_hw_pred")
                                sys.exit(0)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            if self.opt['network_g']['type'].find('SwinIROffsetDenseShift_hw_pred')>-1:
                                if y==0 and x==0:
                                    offset_pred, output_tile = self.net_g.test_forward(input_tile)
                                else:
                                    _, output_tile = self.net_g.test_forward(input_tile, offset_pred, test=True)
                                
                            else:
                                print("only for SwinIROffsetDenseShift_hw_pred")
                                sys.exit(0)
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
            # psnr_b_record[idx] = calculate_psnrb(lq_img,gt_img,border=0,test_y_channel=False)
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

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


