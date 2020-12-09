import argparse
import os
import pdb
import random
import re

import numpy
import torch

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
parser.add_argument('-target', type=str, default='generate')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-npose', type=int, default=8)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_num', type=list, default=5)
opt = parser.parse_args()

def generate(model, dataloader, dataset, path):
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, poses, depths = data
            images, poses, depths = images.cuda(), poses.cuda(), depths.cuda()
            hidden = model(images, poses)
            data = [hidden, depths, poses]
            dataset.store_dataframe(data, batch_idx)
    dataset.store_all(path)


if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)
    os.system('mkdir -p ' + opt.dataset + '/generated_data')

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # define colored_lane symbol for dataloader
    trainset = MRPGDataSet(opt)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # define model file name
    print(f'[will load model: {opt.model_file}]')
    mfile = 'trained_models/' + opt.model_file + '.model'

    # load previous checkpoint or create new model
    checkpoint = torch.load(mfile)
    model = checkpoint['model']
    model.extract_feature = True

    stats = torch.load(opt.dataset + '/data_stats.pth')

    model_num = re.search("camera_num=\d", "model=single_view-bsize=4-lrt=0.01-camera_num=5-seed=1") #
    store_path = opt.dataset + '/generated_data/' + trainset.camera_names[0] + \
                 f'_c{opt.camera_num}m{model_num.group(0)[-1]}.pth'

    model.cuda()
    print('[generating]')
    pdb.set_trace()
    generate(model, trainloader, trainset, store_path)
