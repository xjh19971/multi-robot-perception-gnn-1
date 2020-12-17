import argparse
import cv2
import math
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import utils
from dataloader import MultiViewDGLDataset, SingleViewDataset
from dgl import batch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
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
parser.add_argument('-camera_idx', type=str, default="01234")
parser.add_argument('-apply_noise_idx', type=list, default=None)
parser.add_argument('-model_file', type=str)
parser.add_argument('-visualization', action="store_true")
parser.add_argument('-vis_folder', type=str, default='')
opt = parser.parse_args()
opt.camera_idx = list(map(int,list(opt.camera_idx)))
if opt.apply_noise_idx is not None:
    opt.apply_noise_idx = list(map(int,list(opt.apply_noise_idx)))
opt.camera_num = len(opt.camera_idx)

def _collate_fn(graph):
    return batch(graph)

def visualization(images, gts, preds, stats, batch_idx):
    if opt.apply_noise_idx is not None:
        vis_camera = 1
    else:
        vis_camera = opt.camera_num
    for i in range(vis_camera):
        image = (SingleViewDataset.unormalise_object(images.clone(), stats['images_mean'], stats['images_std'], 'image',
                                                     use_cuda=True)[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1,2,0) * 255.).astype(np.uint8)
        max_depth = stats['max_depth'].numpy()
        gt = gts[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0)
        gt[gt < 0] = 0
        gt[gt > max_depth] = max_depth
        gt = gt / max_depth
        pred = preds[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0)
        pred[pred < 0] = 0
        pred[pred > max_depth] = max_depth
        pred = pred / max_depth
        heatmap_gt = cv2.applyColorMap((gt * 255.).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap((pred * 255.).astype(np.uint8), cv2.COLORMAP_JET)
        plt.imsave('vis_'+opt.vis_folder+'/depth/' + str(i) + str(batch_idx) + '.png', heatmap, cmap='magma', vmax=max_depth)
        plt.imsave('vis_'+opt.vis_folder+'/depth_gt/' + str(i) + str(batch_idx) + '.png', heatmap_gt, cmap='magma', vmax=max_depth)
        plt.imsave('vis_'+opt.vis_folder+'/image/' + str(i) + str(batch_idx) + '.png', image)

def compute_smooth_L1loss(target_depth, predicted_depth, reduction='mean', dataset='airsim-mrmps-data'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
        valid_target = target_depth > 0
    else:
        valid_target = torch.all(torch.cat([target_depth < 100.0,target_depth > 0],dim=1),dim=1,keepdim=True)
    invalid_pred = predicted_depth <= 0
    predicted_depth[invalid_pred] = 1e-8
    loss = F.smooth_l1_loss(predicted_depth[valid_target], target_depth[valid_target], reduction=reduction)
    return loss

def compute_Depth_SILog(target_depth, predicted_depth, lambdad=1.0, dataset='airsim-mrmps-data'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    SILog = 0
    for i in range(len(target_depth)):
        if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
            valid_target = target_depth[i] > 0
        else:
            valid_target = torch.all(torch.cat([target_depth[i] < 100.0,target_depth[i] > 0],dim=0),dim=0,keepdim=True)
        invalid_pred = predicted_depth[i] <= 0
        num_pixels = torch.sum(valid_target)
        predicted_depth[i][invalid_pred] = 1e-8
        distance = torch.log(predicted_depth[i][valid_target]) - torch.log(target_depth[i][valid_target])
        SILog += torch.sum(torch.square(distance)) / num_pixels - torch.square(
            torch.sum(distance)) * lambdad / torch.square(
            num_pixels)
    SILog /= target_depth.size(0)
    return SILog

def compute_Metric(gt, pred, dataset='airsim-mrmps-data'):
    gt = gt.view(-1, 1, opt.image_size, opt.image_size)
    pred = pred.view(-1, 1, opt.image_size, opt.image_size)
    if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
        valid_target = gt > 0
    else:
        valid_target = torch.all(torch.cat([gt < 100.0,gt > 0],dim=1),dim=1,keepdim=True)
    invalid_pred = pred <= 0
    pred[invalid_pred] = 1e-8
    rmse = (gt[valid_target] - pred[valid_target]) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt[valid_target]) - torch.log10(pred[valid_target])) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt[valid_target] - pred[valid_target]) / gt[valid_target])

    sq_rel = torch.mean(((gt[valid_target] - pred[valid_target]) ** 2) / gt[valid_target])
    return abs_rel, sq_rel, rmse, rmse_log

def test(model, dataloader, stats):
    model.eval()
    abs_rel, sq_rel, rmse, rmse_log = 0,0,0,0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths = data
            images, poses, depths = images.cuda(), poses.cuda(), depths.cuda()
            pred_depths = model(images, poses, False)
            if opt.visualization:
                visualization(images, depths, pred_depths, stats, batch_idx)
            # test_loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            # test_loss += compute_Depth_SILog(depths, pred_depth, dataset=opt.dataset)
            abs_rel_single, sq_rel_single, rmse_single, rmse_log_single = compute_Metric(depths,pred_depths,dataset=opt.dataset)
            abs_rel+=abs_rel_single
            sq_rel+=sq_rel_single
            rmse+=rmse_single
            rmse_log+=rmse_log_single
            batch_num += 1
    avg_abs_loss = abs_rel / batch_num
    avg_sq_loss = sq_rel / batch_num
    avg_rmse_loss = rmse / batch_num
    avg_rmse_log_loss = rmse_log / batch_num

    return [avg_abs_loss,avg_sq_loss,avg_rmse_loss,avg_rmse_log_loss]

def test_dgl(model, dataloader, stats, opt):
    model.eval()
    abs_rel, sq_rel, rmse, rmse_log = 0,0,0,0
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.eval()
            data=data.to('cuda:0')
            images = data.ndata['image']
            images = images.view((-1, opt.camera_num, 3, opt.image_size, opt.image_size))
            depths = data.ndata['depth']
            depths  = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            pred_depth = model(data)
            pred_depth = pred_depth.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            if opt.visualization:
                visualization(images, depths, pred_depth, stats, batch_idx)
            #test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=1.0, dataset=opt.dataset)
            #test_loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            batch_num += 1
            abs_rel_single, sq_rel_single, rmse_single, rmse_log_single = compute_Metric(depths,pred_depth,dataset=opt.dataset)
            abs_rel+=abs_rel_single
            sq_rel+=sq_rel_single
            rmse+=rmse_single
            rmse_log+=rmse_log_single
            batch_num += 1
    avg_abs_loss = abs_rel / batch_num
    avg_sq_loss = sq_rel / batch_num
    avg_rmse_loss = rmse / batch_num
    avg_rmse_log_loss = rmse_log / batch_num

    return [avg_abs_loss,avg_sq_loss,avg_rmse_loss,avg_rmse_log_loss]
    # avg_test_loss = test_loss / batch_num
    # return [avg_test_loss]

if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)
    if opt.visualization:
        os.system('mkdir -p ' + 'vis_'+opt.vis_folder)
        os.system('mkdir -p ' + 'vis_'+opt.vis_folder+'/depth')
        os.system('mkdir -p ' + 'vis_'+opt.vis_folder+'/depth_gt')
        os.system('mkdir -p ' + 'vis_'+opt.vis_folder+'/image')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    if opt.dataset=="airsim":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="airsim-noise":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="airsim-dgl":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="airsim-mrmps-data", save_dir="airsim-mrmps-process")
        print(dataset[0])
    elif opt.dataset=="airsim-noise-dgl":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="airsim-mrmps-noise-data", save_dir="airsim-mrmps-noise-process")
        print(dataset[0])
    elif opt.dataset=="cargo":
        opt.dataset = "cargo"
        print(f'[Loading cargo SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-noise-1":
        opt.dataset = "cargo-noise-1"
        print(f'[Loading cargo noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-noise-2":
        opt.dataset = "cargo-noise-2"
        print(f'[Loading cargo noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-noise-3":
        opt.dataset = "cargo-noise-3"
        print(f'[Loading cargo noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="cargo-dgl":
        opt.dataset = "cargo"
        print(f'[Loading cargo MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="cargo", save_dir="cargo-process")
        print(dataset[0])
    elif opt.dataset=="cargo-noise-dgl":
        opt.dataset = "cargo-noise-"+str(len(opt.apply_noise_idx))
        print(f'[Loading cargo noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="cargo-noise-"+str(len(opt.apply_noise_idx)), save_dir="cargo-noise-"+str(len(opt.apply_noise_idx))+"-process")
        print(dataset[0])
    elif opt.dataset=="industrial":
        opt.dataset = "industrial"
        print(f'[Loading industrial SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="industrial-noise-1":
        opt.dataset = "industrial-noise-1"
        print(f'[Loading industrial noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="industrial-noise-2":
        opt.dataset = "industrial-noise-2"
        print(f'[Loading industrial noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset=="industrial-dgl":
        opt.dataset = "industrial"
        print(f'[Loading industrial MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="industrial", save_dir="industrial-process")
        print(dataset[0])
    elif opt.dataset=="industrial-noise-dgl":
        opt.dataset = "industrial-noise-"+str(len(opt.apply_noise_idx))
        print(f'[Loading industrial noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="industrial-noise-"+str(len(opt.apply_noise_idx)), save_dir="industrial-noise-"+str(len(opt.apply_noise_idx))+"-process")
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
    print(str(test_losses))
    # print(utils.format_losses(*test_losses, split='test'))