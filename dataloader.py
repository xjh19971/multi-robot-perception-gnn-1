import argparse
import glob
import math
import os
import random

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class MRPGDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']

        self.img_transforms = transforms.Compose([
            transforms.Resize((self.opt.image_size, self.opt.image_size)),
            transforms.ToTensor(),
        ])
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        if self.opt.dataset == 'airsim-mrmps-data':
            image_path = self.opt.dataset + '/scene/async_rotate_fog_000_clear/'
            depth_path = self.opt.dataset + '/depth_encoded/async_rotate_fog_000_clear/'
            pose_path = self.opt.dataset + '/pose/async_rotate_fog_000_clear/'
        else:
            image_path = self.opt.dataset + '/scene/'
            depth_path = self.opt.dataset + '/depth/'
            pose_path = self.opt.dataset + '/pose/'
        all_data_path = []
        if self.opt.target == 'test':
            self.real_camera_num = len(self.camera_names)
        else:
            self.real_camera_num = self.opt.camera_num
        for i in range(self.real_camera_num):
            all_data_path.append(self.opt.dataset + '/' + self.camera_names[i] + '_all_data.pth')

        self.images = [[] for i in range(self.real_camera_num)]
        self.depths = [[] for i in range(self.real_camera_num)]
        self.poses = [[] for i in range(self.real_camera_num)]

        if not os.path.exists(all_data_path[-1]):
            assert (self.opt.camera_num == 5)
            if self.opt.dataset == 'airsim-mrmps-data':
                image_dirs = next(os.walk(image_path))[1]
                image_dirs.sort()
                depth_dirs = next(os.walk(depth_path))[1]
                depth_dirs.sort()
                pose_dirs = next(os.walk(pose_path))[1]
                pose_dirs.sort()
            else:
                image_dirs = ['']
                depth_dirs = ['']
                pose_dirs = ['']
            for i in range(self.opt.camera_num):
                camera_objects = {}
                for dir_data in image_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data':
                        files_path = image_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = image_path + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        image = Image.open(f'{file_name}')
                        if self.img_transforms is not None:
                            image = self.img_transforms(image)
                            image = image[0:3, :, :]
                        camera_objects[file_name[-10:-4]] = [file_name[-10:-4], image]
                for dir_data in depth_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data':
                        files_path = depth_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = depth_path + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            depth = Image.open(f'{file_name}')
                            if self.opt.dataset == 'airsim-mrmps-data':
                                depth = np.array(
                                    depth[:, :, 0] * (256 ** 3) + depth[:, :, 1] * (256 ** 2) + depth[:, :, 2] * (256 ** 1),
                                    dtype=np.uint32)
                                depth = depth.view(np.float32)
                            else:
                                depth = np.array(depth)
                            depth = cv2.resize(depth, (self.opt.image_size, self.opt.image_size),
                                               interpolation=cv2.INTER_CUBIC)
                            depth = torch.tensor(depth).view(1, self.opt.image_size, self.opt.image_size)
                            camera_objects[file_name[-10:-4]].append(depth)
                for dir_data in pose_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data':
                        files_path = pose_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = pose_path + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.txt')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            with open(f'{file_name}', 'r') as f:
                                pose = f.readlines()
                                pose = list(map(float, pose))
                                pose = torch.tensor(pose)
                                camera_objects[file_name[-10:-4]].append(pose)

                consistent_objects = []
                for k, v in camera_objects.items():
                    if len(v) == 4:
                        consistent_objects.append(v)

                print(f'[Saving {all_data_path[i]} to disk]')
                torch.save(consistent_objects, all_data_path[i])

                del consistent_objects, camera_objects

            print(f'[Reload all data to get valid index]')
            consistent_camera_objects = [[] for i in range(self.opt.camera_num)]
            consistent_camera_id = {}
            for i in range(len(all_data_path)):
                data = torch.load(all_data_path[i])
                for j in range(len(data)):
                    if not data[j][0] in consistent_camera_id:
                        consistent_camera_id[data[j][0]] = [[data[j][1], data[j][2], data[j][3]]]
                    else:
                        consistent_camera_id[data[j][0]].append([data[j][1], data[j][2], data[j][3]])
            for k, v in consistent_camera_id.items():
                if len(v) == self.opt.camera_num:
                    for i in range(self.opt.camera_num):
                        consistent_camera_objects[i].append([v[i][0], v[i][1], v[i][2]])
            for i in range(len(all_data_path)):
                torch.save(consistent_camera_objects[i], all_data_path[i])
            del consistent_camera_objects, consistent_camera_id

        print(f'[Loading all data]')
        for i in range(len(all_data_path)):
            data = torch.load(all_data_path[i])
            for j in range(len(data)):
                self.images[i].append(data[j][0])
                self.depths[i].append(data[j][1])
                self.poses[i].append(data[j][2])

        splits_path = self.opt.dataset + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.n_samples = self.splits.get('n_samples')
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
            if self.opt.target == 'generate':
                self.generated_indx = np.concatenate([self.train_val_indx, self.test_indx])
        else:
            print('[generating data splits]')
            rgn = numpy.random.RandomState(0)
            self.n_samples = len(self.images[0])
            perm = rgn.permutation(self.n_samples)
            n_train_val = int(math.floor(self.n_samples * 0.9))
            self.train_val_indx = perm[0: n_train_val]
            self.test_indx = perm[n_train_val:]
            torch.save(dict(
                n_samples=self.n_samples,
                train_val_indx=self.train_val_indx,
                test_indx=self.test_indx,
            ), splits_path)

        print(f'[Number of samples for each camera: {self.n_samples}]')

        stats_path = self.opt.dataset + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            self.stats = torch.load(stats_path)
        else:
            print('[computing image and depth stats]')
            assert self.opt.camera_num == 5
            stat_images = [[] for i in range(self.opt.camera_num)]
            stat_depths = [[] for i in range(self.opt.camera_num)]
            for i in range(self.opt.camera_num):
                stat_images[i] = torch.stack(self.images[i], dim=0)
                stat_depths[i] = torch.stack(self.depths[i], dim=0)
            stat_images = torch.stack(stat_images, dim=1)
            stat_depths = torch.stack(stat_depths, dim=1)
            self.stats = dict()
            all_images = stat_images.view(-1, 3, stat_images.size(3), stat_images.size(4))
            all_depths = stat_depths.view(-1, 1, stat_depths.size(3), stat_depths.size(4))
            # Compute mean and std for each channel
            self.stats['images_mean'] = torch.mean(all_images, (0, 2, 3))
            self.stats['images_std'] = torch.std(all_images, (0, 2, 3))
            self.stats['depths_mean'] = torch.mean(all_depths, (0, 2, 3))
            self.stats['depths_std'] = torch.std(all_depths, (0, 2, 3))
            torch.save(self.stats, stats_path)

        if self.opt.target == 'generate':
            self.generated_dataset = [[None for i in range(len(self.test_indx) + len(self.train_val_indx))] for j in
                                      range(self.opt.camera_num)]

    def __len__(self):
        if self.opt.target == 'generate':
            return len(self.generated_indx)
        elif self.opt.target == 'test':
            return len(self.test_indx)
        else:
            return len(self.train_val_indx)

    def __getitem__(self, index):
        if self.opt.target == 'test':
            real_index = self.test_indx[index]
        elif self.opt.target == 'generate':
            real_index = self.generated_indx[index]
        else:
            real_index = self.train_val_indx[index]
        # pdb.set_trace()
        image = []
        pose = []
        depth = []
        for i in range(self.real_camera_num):
            image.append(self.images[i][real_index])
            pose.append(self.poses[i][real_index])
            depth.append(self.depths[i][real_index])
        image = torch.stack(image, dim=0)
        pose = torch.stack(pose, dim=0)
        depth = torch.stack(depth, dim=0)
        image = self.normalise_object(image, self.stats['images_mean'], self.stats['images_std'], 'image')
        # depth = self.normalise_object(depth, self.stats['depths_mean'], self.stats['depths_std'], 'depth')
        return image, pose, depth

    def store_dataframe(self, data, idx):
        # pdb.set_trace()
        for i in range(self.opt.camera_num):
            single_data = [data[0][i, :, :, :], data[1][:, i, :, :, :].squeeze(0), data[2][:, i, :].squeeze(0)]
            self.generated_dataset[i][self.generated_indx[idx]] = single_data

    def store_all(self, path, model_num):
        for i in range(len(self.camera_names)):
            real_path = path+self.camera_names[i]+f'_all_data_m{model_num.group(0)[-1]}.pth'
            print(f'[storing feature map at {real_path}]')
            torch.save(self.generated_dataset[i], real_path)

    @staticmethod
    def normalise_object(objects, mean, std, name):
        if name == 'image':
            dim = 3
        else:
            dim = 1
        objects -= mean.view(1, dim, 1, 1)
        objects /= std.view(1, dim, 1, 1)
        return objects

    @staticmethod
    def unormalise_object(objects, mean, std, name, use_cuda=True):
        if name == 'image':
            dim = 3
        else:
            dim = 1
        if use_cuda:
            objects *= mean.view(1, 1, dim, 1, 1).cuda()
            objects += std.view(1, 1, dim, 1, 1).cuda()
        else:
            objects *= mean.view(dim, 1, 1)
            objects += std.view(dim, 1, 1)
        return objects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = argparse.Namespace()
    opt.seed = 1
    opt.dataset = 'airsim-mrmps-data'
    opt.target = 'train'
    opt.batch_size = 1
    dataset = MRPGDataSet(opt)
    trainset, _ = torch.utils.data.random_split(dataset,
                                                [int(0.90 * len(dataset)), len(dataset) - int(0.90 * len(dataset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    a_batch = next(iter(trainloader))
