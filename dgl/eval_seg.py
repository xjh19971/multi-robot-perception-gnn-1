import argparse
import os
import random
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
from dataloader import MultiViewDGLDataset, SingleViewDataset, generate_dataset
from dgl import batch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
dgl_models = ["multi_view_dgl"]
parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim')
parser.add_argument('-task', type=str, default='seg')
parser.add_argument('-target', type=str, default='test')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_idx', type=str, default="01234")
parser.add_argument('-apply_noise_idx', type=list, default=None)
parser.add_argument('-model_file', type=str)
parser.add_argument('-visualization', action="store_true")
parser.add_argument('-vis_folder', type=str, default='')
opt = parser.parse_args()
opt.camera_idx = list(map(int, list(opt.camera_idx)))
if opt.apply_noise_idx is not None:
    opt.apply_noise_idx = list(map(int, list(opt.apply_noise_idx)))
opt.camera_num = len(opt.camera_idx)
cityscapes_map = np.array([
       [0.        , 0.        , 0.        ],
       [0.07843137, 0.07843137, 0.07843137],
       [0.43529412, 0.29019608, 0.        ],
       [0.31764706, 0.        , 0.31764706],
       [0.50196078, 0.25098039, 0.50196078],
       [0.95686275, 0.1372549 , 0.90980392],
       [0.98039216, 0.66666667, 0.62745098],
       [0.90196078, 0.58823529, 0.54901961],
       [0.2745098 , 0.2745098 , 0.2745098 ],
       [0.4       , 0.4       , 0.61176471],
       [0.74509804, 0.6       , 0.6       ],
       [0.70588235, 0.64705882, 0.70588235],
       [0.58823529, 0.39215686, 0.39215686],
       [0.58823529, 0.47058824, 0.35294118],
       [0.6       , 0.6       , 0.6       ],
       [0.6       , 0.6       , 0.6       ],
       [0.98039216, 0.66666667, 0.11764706],
       [0.8627451 , 0.8627451 , 0.        ],
       [0.41960784, 0.55686275, 0.1372549 ],
       [0.59607843, 0.98431373, 0.59607843],
       [0.2745098 , 0.50980392, 0.70588235],
       [0.8627451 , 0.07843137, 0.23529412],
       [1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.55686275],
       [0.        , 0.        , 0.2745098 ],
       [0.        , 0.23529412, 0.39215686],
       [0.        , 0.        , 0.35294118],
       [0.        , 0.        , 0.43137255],
       [0.        , 0.31372549, 0.39215686],
       [0.        , 0.        , 0.90196078],
       [0.46666667, 0.04313725, 0.1254902 ],
       [0.        , 0.        , 0.55686275]])

def _collate_fn(graph):
    return batch(graph)


def visualization_depth(images, gts, preds, stats, batch_idx, user_max_depth=None):
    if opt.apply_noise_idx is not None:
        vis_camera = 1
    else:
        vis_camera = opt.camera_num
    for i in range(vis_camera):
        image = (SingleViewDataset.unormalise_object(images.clone(), stats['images_mean'], stats['images_std'], 'image',
                                                     use_cuda=True)[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255.).astype(np.uint8)
        if user_max_depth is None:
            max_depth = stats['max_depth'].numpy()
            print("Using default max_depth = "+str(max_depth) +" from dataset")
        else:
            max_depth = user_max_depth
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
        plt.imsave('vis_' + opt.vis_folder + '/depth/' + str(i) + str(batch_idx) + '.png', heatmap, cmap='magma',
                   vmax=max_depth)
        plt.imsave('vis_' + opt.vis_folder + '/depth_gt/' + str(i) + str(batch_idx) + '.png', heatmap_gt, cmap='magma',
                   vmax=max_depth)
        plt.imsave('vis_' + opt.vis_folder + '/image/' + str(i) + str(batch_idx) + '.png', image)

def visualization_seg(images, gts, preds, stats, batch_idx):
    if opt.apply_noise_idx is not None:
        vis_camera = 1
    else:
        vis_camera = opt.camera_num
    for i in range(vis_camera):
        image = (SingleViewDataset.unormalise_object(images.clone(), stats['images_mean'], stats['images_std'], 'image',
                                                     use_cuda=True)[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255.).astype(np.uint8)
        gt = gts[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0)
        pred = preds[:, i, :, :, :].cpu().numpy().squeeze(0).transpose(1, 2, 0)
        pred = np.argmax(pred, axis=2)
        segmap_gt = np.array([cityscapes_map[g] for g in gt]).squeeze(2)
        segmap = np.array([cityscapes_map[p] for p in pred])
        plt.imsave('vis_' + opt.vis_folder + '/seg/' + str(i) + str(batch_idx) + '.png', segmap)
        plt.imsave('vis_' + opt.vis_folder + '/seg_gt/' + str(i) + str(batch_idx) + '.png', segmap_gt)
        plt.imsave('vis_' + opt.vis_folder + '/image/' + str(i) + str(batch_idx) + '.png', image)

def compute_meaniou(pr, gt, eps=1e-7):
    intersection = torch.sum(gt * pr, dim=(2,3))
    union = torch.sum(gt, dim=(2,3)) + torch.sum(pr, dim=(2,3)) - intersection + eps
    iou = (intersection + eps) / union
    return torch.mean(iou)

def compute_depth_metric(gt, pred, dataset='airsim-mrmps-data'):
    gt = gt.view(-1, 1, opt.image_size, opt.image_size)
    pred = pred.view(-1, 1, opt.image_size, opt.image_size)
    if dataset == 'airsim-mrmps-data' or dataset == 'airsim-mrmps-noise-data':
        valid_target = gt > 0
    else:
        valid_target = torch.all(torch.cat([gt < 100.0, gt > 0], dim=1), dim=1, keepdim=True)
    invalid_pred = pred <= 0
    pred[invalid_pred] = 1e-8
    rmse = (gt[valid_target] - pred[valid_target]) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt[valid_target]) - torch.log10(pred[valid_target])) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt[valid_target] - pred[valid_target]) / gt[valid_target])

    sq_rel = torch.mean(((gt[valid_target] - pred[valid_target]) ** 2) / gt[valid_target])
    return abs_rel, sq_rel, rmse, rmse_log

def compute_seg_metric(gt, pred, output_dim, activation=None):
    gt = one_hot(gt, output_dim)
    gt = gt.view(-1, output_dim, opt.image_size, opt.image_size)
    pred = pred.view(-1, output_dim, opt.image_size, opt.image_size)
    if activation == 'softmax':
        pred = torch.nn.Softmax(dim=1)(pred)
    pr = torch.argmax(pred, dim=1)
    pr = one_hot(pr, output_dim).transpose(2,3).transpose(1,2)
    meaniou = compute_meaniou(pr, gt)
    return meaniou

def test(model, dataloader, stats, opt):
    model.eval()
    abs_rel, sq_rel, rmse, rmse_log, mean_iou = 0, 0, 0, 0, 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths, segs = data
            images, poses, depths, segs = images.cuda(), poses.cuda(), depths.cuda(), segs.cuda()
            if opt.task == 'depth':
                pred_depths = model(images)
            elif opt.task == 'seg':
                pred_segs = model(images)
            if opt.visualization:
                if opt.task=='depth':
                    visualization_depth(images, depths, pred_depths, stats, batch_idx)
                elif opt.task == 'seg':
                    visualization_seg(images, segs, pred_segs, stats, batch_idx)
            # test_loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            # test_loss += compute_Depth_SILog(depths, pred_depth, dataset=opt.dataset)
            if opt.task=='depth':
                abs_rel_single, sq_rel_single, rmse_single, rmse_log_single = compute_depth_metric(depths, pred_depths,
                                                                          dataset=opt.dataset)
                abs_rel += abs_rel_single
                sq_rel += sq_rel_single
                rmse += rmse_single
                rmse_log += rmse_log_single
            elif opt.task == 'seg':
                mean_iou_single = compute_seg_metric(segs, pred_segs, opt.output_dim)
                mean_iou += mean_iou_single
            batch_num += 1
    avg_abs_loss = abs_rel / batch_num
    avg_sq_loss = sq_rel / batch_num
    avg_rmse_loss = rmse / batch_num
    avg_rmse_log_loss = rmse_log / batch_num
    avg_mean_iou = mean_iou / batch_num
    return [avg_abs_loss, avg_sq_loss, avg_rmse_loss, avg_rmse_log_loss, avg_mean_iou]


def test_dgl(model, dataloader, stats, opt):
    model.eval()
    abs_rel, sq_rel, rmse, rmse_log, mean_iou = 0, 0, 0, 0, 0
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.eval()
            data = data.to('cuda:0')
            images = data.ndata['image']
            images = images.view((-1, opt.camera_num, 3, opt.image_size, opt.image_size))
            depths = data.ndata['depth']
            depths = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            segs = data.ndata['seg']
            segs = segs.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            if opt.task == 'depth':
                pred_depth = model(data)
                pred_depth = pred_depth.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            elif opt.task == 'seg':
                pred_seg = model(data)
                pred_seg = pred_seg.view((-1, opt.camera_num, opt.output_dim, opt.image_size, opt.image_size))
            if opt.visualization:
                if opt.task=='depth':
                    visualization_depth(images, depths, pred_depth, stats, batch_idx)
                elif opt.task == 'seg':
                    visualization_seg(images, segs, pred_seg, stats, batch_idx)
            # test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=1.0, dataset=opt.dataset)
            # test_loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            batch_num += 1
            if opt.task == 'depth':
                abs_rel_single, sq_rel_single, rmse_single, rmse_log_single = compute_depth_metric(depths, pred_depth,
                                                                                         dataset=opt.dataset)
                abs_rel += abs_rel_single
                sq_rel += sq_rel_single
                rmse += rmse_single
                rmse_log += rmse_log_single
            elif opt.task == 'seg':
                mean_iou_single = compute_seg_metric(segs, pred_seg, opt.output_dim)
                mean_iou += mean_iou_single
            batch_num += 1
    avg_abs_loss = abs_rel / batch_num
    avg_sq_loss = sq_rel / batch_num
    avg_rmse_loss = rmse / batch_num
    avg_rmse_log_loss = rmse_log / batch_num
    avg_mean_iou = mean_iou / batch_num
    return [avg_abs_loss, avg_sq_loss, avg_rmse_loss, avg_rmse_log_loss, avg_mean_iou]


if __name__ == '__main__':
    if opt.visualization:
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder)
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder + '/depth')
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder + '/depth_gt')
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder + '/seg')
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder + '/seg_gt')
        os.system('mkdir -p ' + 'vis_' + opt.vis_folder + '/image')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    dataset = generate_dataset(opt)

    if opt.model in dgl_models:
        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.batch_size, collate_fn=_collate_fn)
    else:
        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.batch_size)
    print(f'[will load model: {opt.model_file}]')
    print(f'[testing camera idx: {opt.camera_idx}]')
    mfile = opt.model_file + '.model'
    opt.output_dim = int(dataset.stats['num_classes']) + 1
    checkpoint = torch.load(opt.model_dir + '/' + mfile)
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
        test_losses = test(model, testloader, dataset.stats, opt)
    elif opt.model in dgl_models:
        test_losses = test_dgl(model, testloader, dataset.stats, opt)
    t1 = time.time()
    print("Time per epoch= %d s" % (t1 - t0))
    # print(utils.format_losses(*test_losses, split='test'))
    print(str(test_losses))

