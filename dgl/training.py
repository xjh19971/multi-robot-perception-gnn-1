import argparse
import math
import os
import random
import time

import numpy
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
from dataloader import generate_dataset
from model import models

from dgl import batch

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

dgl_models = ["multi_view_dgl"]
parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim')
parser.add_argument('-task', type=str, default='depth')
parser.add_argument('-target', type=str, default='train')
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.005)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_idx', type=str, default="01234")
parser.add_argument('-eval_camera_idx', type=str, default="01234")
parser.add_argument('-pretrained', action="store_true", default=True)
parser.add_argument('-multi_gpu', action="store_true")
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-apply_noise_idx', type=str, default=None)
parser.add_argument('-model_file', type=str, default=None)
parser.add_argument('-backbone', type=str, default='resnet50')
parser.add_argument('-skip_level', action="store_true")
parser.add_argument('-multi_gcn', action="store_true")
parser.add_argument('-compress_gcn', action="store_true")
parser.add_argument('-lambda_edge', type=float, default=1e-0)
parser.add_argument('-gpu_idx', type=str, default="0")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx
opt.camera_idx = list(map(int, list(opt.camera_idx)))
if opt.apply_noise_idx is not None:
    opt.apply_noise_idx = list(map(int, list(opt.apply_noise_idx)))
opt.eval_camera_idx = list(map(int, list(opt.eval_camera_idx)))
opt.camera_num = len(opt.camera_idx)
opt.eval_camera_num = len(opt.eval_camera_idx)


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

def compute_edge_aware_loss(disp, img, dgl=False):
    """
    From https://github.com/nianticlabs/monodepth2/
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.clone().mean(3, True).mean(4, True)
    norm_disp = disp / (mean_disp + 1e-7)
    grad_disp_x = torch.abs(norm_disp[:, :, :, :, :-1] - norm_disp[:, :, :, :, 1:])
    grad_disp_y = torch.abs(norm_disp[:, :, :, :-1, :] - norm_disp[:, :, :, 1:, :])

    if dgl:
        temp_img = img.clone().view(img.size(0) // 5, 5, 3, img.size(2), img.size(3))
    else:
        temp_img = img
    grad_img_x = torch.mean(torch.abs(temp_img[:, :, :, :, :-1] - temp_img[:, :, :, :, 1:]), 2, keepdim=True)
    grad_img_y = torch.mean(torch.abs(temp_img[:, :, :, :-1, :] - temp_img[:, :, :, 1:, :]), 2, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_cross_entropy2d(target_seg, predicted_seg, reduction='mean', output_dim=1, weight=None):
    target_seg = target_seg.view(-1, opt.image_size, opt.image_size)
    predicted_seg = predicted_seg.view(-1, output_dim, opt.image_size, opt.image_size)
    loss = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)(predicted_seg, target_seg)
    return loss

def train(model, dataloader, optimizer, epoch, stats, log_interval=50, lambda_edge=0, task='seg'):
    model.train()
    train_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        images, poses, depths, segs= data
        images, poses, depths, segs = images.cuda(), poses.cuda(), depths.cuda(), segs.cuda()

        pred_depth = None
        pred_seg = None
        if task=='depth':
            pred_depth = model(images)
        elif task=='seg':
            pred_seg = model(images)
        elif task=='depthseg':
            pred_depth, pred_seg = model(images)

        loss = 0
        if pred_depth is not None:
            loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            edge_loss = compute_edge_aware_loss(pred_depth, images)
            loss += lambda_edge * edge_loss
        if pred_seg is not None:
            loss += compute_cross_entropy2d(segs, pred_seg, output_dim=opt.output_dim)

        train_loss += loss

        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            optimizer.step()
        batch_num += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * opt.batch_size, len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))

    avg_train_loss = train_loss / batch_num
    return [avg_train_loss]


def test(model, dataloader, stats, lambda_edge=0, task='seg'):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths, segs = data
            images, poses, depths, segs = images.cuda(), poses.cuda(), depths.cuda(), segs.cuda()

            pred_depth = None
            pred_seg = None
            if task == 'depth':
                pred_depth = model(images)
            elif task == 'seg':
                pred_seg = model(images)
            elif task == 'depthseg':
                pred_depth, pred_seg = model(images)

            loss = 0
            if pred_depth is not None:
                loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
                edge_loss = compute_edge_aware_loss(pred_depth, images)
                loss += lambda_edge * edge_loss
            if pred_seg is not None:
                loss += compute_cross_entropy2d(segs, pred_seg, output_dim=opt.output_dim)

            test_loss += loss
            batch_num += 1

    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]


def train_dgl(model, dataloader, optimizer, epoch, stats, opt, log_interval=50, lambda_edge=0, task='seg'):
    model.train()
    train_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        model.train()
        data = data.to('cuda:0')
        depths = data.ndata['depth']
        depths = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
        segs = data.ndata['seg']
        segs = segs.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))

        pred_depth = None
        pred_seg = None
        if task == 'depth':
            pred_depth = model(data)
        elif task == 'seg':
            pred_seg = model(data)
        elif task=='depthseg':
            pred_depth, pred_seg = model(data)

        loss = 0
        if pred_depth is not None:
            loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
            edge_loss = compute_edge_aware_loss(pred_depth, data.ndata['image'], dgl=True)
            loss += lambda_edge * edge_loss
        if pred_seg is not None:
            loss += compute_cross_entropy2d(segs, pred_seg, output_dim=opt.output_dim)

        train_loss += loss

        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            optimizer.step()
        batch_num += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * opt.batch_size, len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))

    avg_train_loss = train_loss / batch_num
    return [avg_train_loss]


def test_dgl(model, dataloader, stats, opt, lambda_edge=0, task='seg'):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.eval()
            data = data.to('cuda:0')
            depths = data.ndata['depth']
            depths = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            segs = data.ndata['seg']
            segs = segs.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
            pred_depth = None
            pred_seg = None
            if task == 'depth':
                pred_depth = model(data)
            elif task == 'seg':
                pred_seg = model(data)
            elif task == 'depthseg':
                pred_depth, pred_seg = model(data)

            loss = 0
            if pred_depth is not None:
                loss += compute_smooth_L1loss(depths, pred_depth, dataset=opt.dataset)
                edge_loss = compute_edge_aware_loss(pred_depth, data.ndata['image'], dgl=True)
                loss += lambda_edge * edge_loss
            if pred_seg is not None:
                loss += compute_cross_entropy2d(segs, pred_seg, output_dim=opt.output_dim)

            test_loss += loss
            batch_num += 1
    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]


if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    dataset = generate_dataset(opt)

    trainset, valset = torch.utils.data.random_split(dataset, [int(0.90 * len(dataset)),
                                                               len(dataset) - int(0.90 * len(dataset))])
    if opt.model in dgl_models:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                                collate_fn=_collate_fn)
        valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False,
                                                collate_fn=_collate_fn)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                                 )
        valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False,
                                                )

    # define model file name
    print(f'[will save model as: {opt.model_file}]')
    mfile = opt.model_file + '.model'
    if opt.task == 'seg':
        opt.output_dim = int(dataset.stats['num_classes']) + 1
    elif opt.task == 'depth':
        opt.output_dim = 1
    elif opt.task == 'depthseg':
        opt.output_dim = int(dataset.stats['num_classes']) + 1

    # load previous checkpoint or create new model
    if os.path.isfile(opt.model_dir + '/' + mfile):
        print(f'[loading previous checkpoint: {mfile}]')
        checkpoint = torch.load(opt.model_dir + '/' + mfile)
        model = checkpoint['model']
        model.cuda()
        optimizer = optim.Adam(model.parameters(), opt.lrt)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40)
        scheduler.load_state_dict(checkpoint['scheduler'])
        n_iter = checkpoint['n_iter']
        utils.log(opt.model_dir + '/' + opt.model_file + '.log', '[resuming from checkpoint]')
    else:
        model = None
        if opt.backbone == 'mobilenetv2':
            opt.feature_dim = 1280
        elif opt.backbone == 'resnet50':
            opt.feature_dim = 2048
        elif opt.backbone == 'resnet18':
            opt.feature_dim = 512
        assert hasattr(opt, 'feature_dim')
        if opt.model == "single_view":
            model = models.single_view_model(opt)
        elif opt.model == "multi_view":
            model = models.multi_view_model(opt)
        elif opt.model == "multi_view_dgl":
            model = models.multi_view_dgl_model(opt)
        assert model is not None

        optimizer = optim.Adam(model.parameters(), opt.lrt)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40)
        n_iter = 0
    # if opt.model in ["single_view", "multi_view"]:
    #     stats = torch.load(opt.dataset + '/data_stats.pth')
    # elif opt.model in dgl_models:
    #     stats = torch.load(dataset.save_dir + '/data_stats-'+str(opt.camera_idx)+'.pth')
    if opt.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    print('[training]')
    print('[Batch size]: ', opt.batch_size)
    min_val_loss = 1000000000
    for epoch in range(opt.epoch):
        t0 = time.time()
        if opt.model == "single_view":
            train_losses = train(model, trainloader, optimizer, epoch, dataset.stats, lambda_edge=opt.lambda_edge, task=opt.task)
        elif opt.model in dgl_models:
            train_losses = train_dgl(model, trainloader, optimizer, epoch, dataset.stats, opt, lambda_edge=opt.lambda_edge, task=opt.task)
        t1 = time.time()
        print("Time per epoch= %d s" % (t1 - t0))
        if opt.model == "single_view":
            val_losses = test(model, valloader, dataset.stats, task=opt.task)
        elif opt.model in dgl_models:
            val_losses = test_dgl(model, valloader, dataset.stats, opt, task=opt.task)
        scheduler.step(val_losses[0])
        if val_losses[0] < min_val_loss:
            torch.save({'model': model,
                        'optimizer': optimizer.state_dict(),
                        'n_iter': n_iter,
                        'scheduler': scheduler.state_dict()}, opt.model_dir + '/best-' + mfile)
            min_val_loss = val_losses[0]
        n_iter += 1
        model.cpu()
        torch.save({'model': model,
                    'optimizer': optimizer.state_dict(),
                    'n_iter': n_iter,
                    'scheduler': scheduler.state_dict()}, opt.model_dir + '/' + mfile)
        model.cuda()
        log_string = f'step {n_iter} | '
        log_string += utils.format_losses(*train_losses, split='train', task=opt.task)
        log_string += utils.format_losses(*val_losses, split='valid', task=opt.task)
        print(log_string)
        utils.log(opt.model_dir + '/' + opt.model_file + '.log', log_string)
