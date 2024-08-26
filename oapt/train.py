# flake8: noqa
import os.path as osp

import sys
basicsr_path = "/root/miniconda3/envs/hat/lib/python3.8/site-packages/basicsr/"
hat_path = "/home/moqiao/workplace/HAT-main"
sys.path.insert(0, basicsr_path)
sys.path.insert(0, hat_path)

import oapt.archs #oapt.
import oapt.data
import oapt.models
import oapt.losses
from basicsr.train import train_pipeline
import warnings
import os
warnings.filterwarnings("ignore")
# os.environ['RANK']='4'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
# os.environ['WORLD_SIZE']




if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
