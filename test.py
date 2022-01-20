from src.model import LinearModel, weight_init
import os
import sys
import pdb
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pdb
import matplotlib.gridspec as gridspec
import src.misc as misc
from src.datasets.human36m import Human36M
from collections import OrderedDict
from src.data_process import unNormalizeData
from src.camera import *
import matplotlib.pyplot as plt
from src.vis import *
cam2id = {
  "54138969":1,
  "55011271":2,
  "58860488":3,
  "60457274":4
}
#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='1' python3 test.py --b 512 --test_file ./shell/temp_bak.txt --checkpoint ./output/2021-08-24-11-26/model_best.pth 
def parse_arg():
    parser = argparse.ArgumentParser(description="test car")
    parser.add_argument('--b', help='Batch size ', type=int,default=1)
    parser.add_argument('--checkpoint', help='resume checkpoint .pth', type=str,default="./checkpoint/ckpt_best.pth.tar")
    parser.add_argument('--test_dir', help='test dir', type=str,default="./data")
    args = parser.parse_args()
    print(args)
    return args

if __name__ =="__main__":
    args = parse_arg()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    #define model
    model = LinearModel()
    #load weight
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    #load test data
    rcams = load_cameras()
    actions=misc.define_actions("All")
    test_3d = torch.load(os.path.join(args.test_dir, 'test_3d.pth.tar'))
    test_2d = torch.load(os.path.join(args.test_dir, 'test_2d.pth.tar'))
    #load mean/std file
    stat_3d = torch.load(os.path.join(args.test_dir, 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join(args.test_dir, 'stat_2d.pth.tar'))
    for k2d in test_2d.keys():
        (sub, act, fname) = k2d
        if act not in actions:
            continue
        k3d = k2d
        k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
        camid = cam2id[k3d[-1].split('.')[1]]
        #get R/T
        R, T, f, c, k, p, name=rcams[(k3d[0],camid)]
        #帧数
        num_f, _ = test_2d[k2d].shape
        assert test_2d[k2d].shape[0] == test_3d[k3d].shape[0]
        for i in range(0,num_f,1000):
            inps = test_2d[k2d][i]
            gt = test_3d[k3d][i]
            inputs = Variable(torch.tensor(inps.reshape(1,32)).to(torch.float32).to(device))
            targets = Variable(torch.tensor(gt.reshape(1,48)).to(torch.float32).to(device))
            #forward,[b,48],unnormal
            outputs = model(inputs)
            p3d = unNormalizeData(outputs.cpu().detach().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
            p2d = unNormalizeData(inps.reshape(1,32), stat_2d['data_mean_2d'], stat_2d['data_std_2d'], stat_2d['dim_to_use_2d'])
            gt = unNormalizeData(gt,stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
            #centered
            p3d = cam2world_centered(p3d,R,T)
            gt = cam2world_centered(gt,R,T)
            #vis
            ## 1080p	= 640 x 320
            fig = plt.figure( figsize=(6.4, 3.2) )
            gs1 = gridspec.GridSpec(1, 3) # 5 rows, 9 columns
            gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
            plt.axis('off')
            #show 2d
            ax1 = plt.subplot(gs1[0])
            show2Dpose(p2d, ax1)
            ax1.invert_yaxis()
            #show 3d_gt 
            ax2 = plt.subplot(gs1[1], projection='3d')
            gt= gt[0,:]
            show3Dpose(gt, ax2)
            #show pred 3d
            ax3 = plt.subplot(gs1[2], projection='3d')
            p3d = p3d[0,:]
            show3Dpose(p3d, ax3,lcolor="#9b59b6", rcolor="#2ecc71")
            plt.savefig(str(act)+str(i)+"_filename.png")
            plt.close()
