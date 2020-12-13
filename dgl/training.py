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
from dataloader import AirsimMapDGLDataset, MRPGDataSet
from model import models, blocks
from dgl import batch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim-mrmps-data')
parser.add_argument('-target', type=str, default='train')
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.005)
parser.add_argument('-npose', type=int, default=8)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_num', type=int, default=5)
parser.add_argument('-pretrained', action="store_true", default=True)
parser.add_argument('-multi_gpu', action="store_true")
parser.add_argument('-epoch', type=int, default=200)
opt = parser.parse_args()

def _collate_fn(graph):
    return batch(graph)


def compute_MSE_loss(target_depth, predicted_depth, reduction='mean'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    loss = F.mse_loss(predicted_depth, target_depth, reduction=reduction)
    return loss

def compute_smooth_L1loss(target_depth, predicted_depth, reduction='mean'):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    loss = F.smooth_l1_loss(predicted_depth, target_depth, reduction=reduction)
    return loss

def compute_Depth_SILog(target_depth, predicted_depth, lambdad=0.0):
    target_depth = target_depth.view(-1, 1, opt.image_size, opt.image_size)
    predicted_depth = predicted_depth.view(-1, 1, opt.image_size, opt.image_size)
    SILog = 0
    for i in range(len(target_depth)):
        valid_target = target_depth[i] > 0
        invalid_pred = predicted_depth[i] <= 0
        num_pixels = torch.sum(valid_target)
        predicted_depth[i][invalid_pred] = 1e-8
        distance = torch.log(predicted_depth[i][valid_target]) - torch.log(target_depth[i][valid_target])
        SILog += torch.sum(torch.square(distance)) / num_pixels - torch.square(
            torch.sum(distance)) * lambdad / torch.square(
            num_pixels)
    SILog /= target_depth.size(0)
    return SILog


def train(model, dataloader, optimizer, epoch, stats, log_interval=50):
    model.train()
    train_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        images, poses, depths = data
        images, poses, depths = images.cuda(), poses.cuda(), depths.cuda()
        pred_depth = model(images, poses, False)
        # loss = compute_MSE_loss(depths, pred_depth)
        # loss = compute_Depth_SILog(depths, pred_depth, lambdad=0.5)
        loss = compute_smooth_L1loss(depths, pred_depth)
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


def test(model, dataloader, stats):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths = data
            images, poses, depths = images.cuda(), poses.cuda(), depths.cuda()
            pred_depth = model(images, poses, False)
            # test_loss += compute_MSE_loss(depths, pred_depth)
            # test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=0.0)
            test_loss += compute_smooth_L1loss(depths, pred_depth)
            batch_num += 1
    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]


def train_dgl(model, dataloader, optimizer, epoch, stats, opt, log_interval=50):
    model.train()
    train_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        model.train()
        data = data.to('cuda:0')
        pred_depth = model(data)
        depths = data.ndata['depth']        
        depths  = depths.view((-1, opt.camera_num, 1, opt.image_size, opt.image_size))
        loss = compute_smooth_L1loss(depths, pred_depth)
        train_loss += loss
        if not math.isnan(loss.item()):
            loss.backward()
            optimizer.step()
        batch_num += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * opt.batch_size, len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
    avg_train_loss = train_loss / batch_num
    return [avg_train_loss]


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
            test_loss += compute_smooth_L1loss(depths, pred_depth)
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
        print(f'[Loading MRPGDataset]')
        dataset = MRPGDataSet(opt)
    elif opt.dataset=="airsim-dgl":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading AirsimMapDGLDataset]')
        dataset = AirsimMapDGLDataset(opt)
        print(dataset[0])

    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [int(0.90 * len(dataset)),
                                                      len(dataset) - int(0.90 * len(dataset))])
    if opt.model == "multi_view_dgl":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size,collate_fn=_collate_fn)
        valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size,collate_fn=_collate_fn)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size)
        valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size)

    # define model file name
    opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-lrt={opt.lrt}-camera_num={opt.camera_num}'
    opt.model_file += f'-seed={opt.seed}'
    print(f'[will save model as: {opt.model_file}]')
    mfile = opt.model_file + '.model'

    # load previous checkpoint or create new model
    if os.path.isfile(mfile):
        print(f'[loading previous checkpoint: {mfile}]')
        checkpoint = torch.load(mfile)
        model = checkpoint['model']
        model.cuda()
        optimizer = optim.Adam(model.parameters(), opt.lrt)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40)
        scheduler.load_state_dict(checkpoint['scheduler'])
        n_iter = checkpoint['n_iter']
        utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
    else:
        if opt.model == "single_view":
            model = models.single_view_model(opt)
        elif opt.model == "multi_view":
            model = models.multi_view_model(opt)
        elif opt.model == "multi_view_dgl":
            model = models.multi_view_dgl_model(opt)

        optimizer = optim.Adam(model.parameters(), opt.lrt)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40)
        n_iter = 0
    if opt.model in ["single_view", "multi_view"]:
        stats = torch.load(opt.dataset + '/data_stats.pth')
    elif opt.model == "multi_view_dgl" :
        stats = torch.load(dataset.save_dir + '/data_stats-'+str(opt.camera_num)+'.pth')
    if opt.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    print('[training]')
    print('[Batch size]: ', opt.batch_size)
    for epoch in range(opt.epoch):
        t0 = time.time()
        if opt.model == "single_view":
            train_losses = train(model, trainloader, optimizer, epoch, dataset.stats)
        elif opt.model == "multi_view_dgl":
            train_losses = train_dgl(model, trainloader, optimizer, epoch, dataset.stats, opt)
        t1 = time.time()
        print("Time per epoch= %d s" % (t1 - t0))
        if opt.model == "single_view":
            val_losses = test(model, valloader, dataset.stats)
        elif opt.model == "multi_view_dgl":
            val_losses = test_dgl(model, valloader, dataset.stats, opt)
        scheduler.step(val_losses[0])
        n_iter += 1
        model.cpu()
        torch.save({'model': model,
                    'optimizer': optimizer.state_dict(),
                    'n_iter': n_iter,
                    'scheduler': scheduler.state_dict()}, opt.model_file + '.model')
        model.cuda()
        log_string = f'step {n_iter} | '
        log_string += utils.format_losses(*train_losses, split='train')
        log_string += utils.format_losses(*val_losses, split='valid')
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)
