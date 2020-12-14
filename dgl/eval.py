import argparse
import math
import os
import random
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from dataloader import MultiViewDGLDataset, SingleViewDataset
from model import models, blocks
from dgl import batch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
dgl_models = ["multi_view_dgl"]
parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim')
parser.add_argument('-target', type=str, default='test')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-npose', type=int, default=8)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_idx', type=list, default=[0,1,2,3,4])
parser.add_argument('-apply_noise_idx', type=list, default=None)
parser.add_argument('-model_file', type=str, default=None)
opt = parser.parse_args()
opt.camera_num = len(opt.camera_idx)

def _collate_fn(graph):
    return batch(graph)

def compute_smooth_L1loss(target_depth, predicted_depth, reduction='mean', dataset='airsim-mrmps-data'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
        valid_target = target_depth > 0
    else:
        valid_target = target_depth < 100.0
    invalid_pred = predicted_depth <= 0
    predicted_depth[invalid_pred] = 1e-8
    loss = F.smooth_l1_loss(predicted_depth[valid_target], target_depth[valid_target], reduction=reduction)
    return loss

def compute_Depth_SILog(target_depth, predicted_depth, lambdad=0.0, dataset='airsim-mrmps-data'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    SILog = 0
    for i in range(len(target_depth)):
        if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
            valid_target = target_depth[i] > 0
        else:
            valid_target = target_depth[i] < 100.0
        invalid_pred = predicted_depth[i] <= 0
        num_pixels = torch.sum(valid_target)
        predicted_depth[i][invalid_pred] = 1e-8
        distance = torch.log(predicted_depth[i][valid_target]) - torch.log(target_depth[i][valid_target])
        SILog += torch.sum(torch.square(distance)) / num_pixels - torch.square(
            torch.sum(distance)) * lambdad / torch.square(
            num_pixels)
    SILog /= target_depth.size(0)
    return SILog

def test(model, dataloader, stats):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths = data
            images, poses, depths = images.cuda(), poses.cuda(), depths.cuda()
            pred_depth = model(images, poses, False)
            test_loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            # test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=0.0, dataset=opt.dataset)
            batch_num += 1
    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]

def test_dgl(model, dataloader, stats, opt):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.eval()
            data=data.to('cuda:0')
            pred_depth = model(data)
            depths = data.ndata['depth']
            depths  = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=0.0, dataset=opt.dataset)
            batch_num += 1
    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]

if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    if opt.dataset=="airsim":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="airsim-noise":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="airsim-dgl":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="airsim-noise-dgl":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="cargo":
        opt.dataset = "cargo"
        print(f'[Loading cargo SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-noise":
        opt.dataset = "cargo-noise"
        print(f'[Loading cargo noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-dgl":
        opt.dataset = "cargo"
        print(f'[Loading cargo MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="cargo-noise-dgl":
        opt.dataset = "cargo-noise"
        print(f'[Loading cargo noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="industrial":
        opt.dataset = "industrial"
        print(f'[Loading industrial SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="industrial-noise":
        opt.dataset = "industrial-noise"
        print(f'[Loading industrial noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="industrial-dgl":
        opt.dataset = "industrial"
        print(f'[Loading industrial MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])
    elif opt.dataset=="industrial-noise-dgl":
        opt.dataset = "industrial-noise"
        print(f'[Loading industrial noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt)
        print(dataset[0])

    if opt.model in dgl_models:
        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size,collate_fn=_collate_fn)
    else:
        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size)

    # define model file name
    #opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-lrt={opt.lrt}-camera_idx={opt.camera_idx}'
    #opt.model_file += f'-seed={opt.seed}'
    print(f'[will load model: {opt.model_file}]')
    print(f'[testing camera idx: {opt.camera_idx}]')
    mfile = opt.model_file + '.model'
    checkpoint = torch.load(opt.model_dir+'/'+mfile)
    model = checkpoint['model']
    # if opt.model in ["single_view", "multi_view"]:
    #     stats = torch.load(opt.dataset + '/data_stats.pth')
    # elif opt.model in dgl_models:
    #     stats = torch.load(dataset.save_dir + '/data_stats-'+str(opt.camera_idx)+'.pth')
    model.cuda()

    print('[testing]')
    print('[Batch size]: ', opt.batch_size)
    t0 = time.time()
    if opt.model == "single_view":
        test_losses = test(model, testloader, dataset.stats)
    elif opt.model in dgl_models:
        test_losses = test_dgl(model, testloader, dataset.stats, opt)
    t1 = time.time()
    print("Time per epoch= %d s" % (t1 - t0))
    print(utils.format_losses(*test_losses, split='test'))