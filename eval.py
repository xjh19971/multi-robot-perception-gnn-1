import argparse
import numpy
import os
import random
import time
import torch

import utils
from dataloader import MRPGDataSet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim-mrmps-data')
parser.add_argument('-model_file', type=str, default='model=single_view-bsize=4-lrt=0.01-camera_num=5-seed=1')
parser.add_argument('-target', type=str, default='test')
parser.add_argument('-batch_size', type=int, default=2)
parser.add_argument('-lrt', type=float, default=0.01)
parser.add_argument('-npose', type=int, default=8)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_num', type=int, default=5)
opt = parser.parse_args()

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
            test_loss += compute_Depth_SILog(depths, pred_depth, lambdad=0.0)
            batch_num += 1
    avg_test_loss = test_loss / batch_num
    return [avg_test_loss]

if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # define colored_lane symbol for dataloader
    testset = MRPGDataSet(opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # define model file name
    print(f'[will load model: {opt.model_file}]')
    mfile = opt.model_file + '.model'

    # load previous checkpoint or create new model
    checkpoint = torch.load('trained_models/'+mfile)
    model = checkpoint['model']

    stats = torch.load(opt.dataset + '/data_stats.pth')
    model.opt.camera_num = 5
    model.cuda()
    print('[testing]')
    t0 = time.time()
    test_losses = test(model, testloader, testset.stats)
    t1 = time.time()
    print("Time per epoch= %d s" % (t1 - t0))
    print(utils.format_losses(*test_losses, split='test'))
