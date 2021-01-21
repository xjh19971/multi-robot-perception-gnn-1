import glob
import math
import os
import random

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from imagenet_c import gaussian_noise, shot_noise, impulse_noise, motion_blur, snow, jpeg_compression
from torchvision import transforms


def load_raw_data(opt, camera_names, img_transforms):
    if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
        image_path = opt.dataset + '/scene/async_rotate_fog_000_clear/'
        depth_path = opt.dataset + '/depth_encoded/async_rotate_fog_000_clear/'
        pose_path = opt.dataset + '/pose/async_rotate_fog_000_clear/'
    else:
        image_path = opt.dataset + '/scene/'
        depth_path = opt.dataset + '/depth_encoded/'
        pose_path = opt.dataset + '/pose/'
    all_data_path = []
    if opt.target == 'test':
        real_camera_num = len(camera_names)
    else:
        real_camera_num = opt.camera_num
    for i in opt.camera_idx:
        all_data_path.append(opt.dataset + '/' + camera_names[i] + '_all_data.pth')

    images = [[] for i in range(real_camera_num)]
    depths = [[] for i in range(real_camera_num)]
    poses = [[] for i in range(real_camera_num)]
    severities = [[] for i in range(real_camera_num)]

    if not os.path.exists(all_data_path[-1]):
        assert (opt.camera_num == 5)
        if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
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
        for i in range(opt.camera_num):
            camera_objects = {}
            severity_dict = {}
            for dir_data in image_dirs:
                if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
                    files_path = image_path + dir_data + '/' + camera_names[i]
                else:
                    files_path = image_path + camera_names[i]
                file_names = glob.glob(f'{files_path}/*.png')
                file_names.sort()
                for file_name in file_names:
                    severity = -1
                    image = Image.open(f'{file_name}')
                    if img_transforms is not None:
                        if opt.apply_noise_idx is not None and i in opt.apply_noise_idx:
                            image, severity = apply_noise_on_images(image, opt)
                        else:
                            image = img_transforms(image)
                        image = image[0:3, :, :]
                    else:
                        print("Default img_transform is not defined!")
                        raise KeyError
                    camera_objects[file_name[-10:-4]] = [file_name[-10:-4], image]
                    severity_dict[file_name[-10:-4]] = severity
            for dir_data in depth_dirs:
                if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
                    files_path = depth_path + dir_data + '/' + camera_names[i]
                else:
                    files_path = depth_path + camera_names[i]
                if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
                    file_names = glob.glob(f'{files_path}/*.png')
                else:
                    file_names = glob.glob(f'{files_path}/*.npy')
                file_names.sort()
                for file_name in file_names:
                    if file_name[-10:-4] in camera_objects:
                        if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
                            depth = cv2.imread(f'{file_name}')
                            depth = np.array(
                                depth[:, :, 0] * (256 ** 3) + depth[:, :, 1] * (256 ** 2) + depth[:, :, 2] * (
                                        256 ** 1),
                                dtype=np.uint32)
                            depth = depth.view(np.float32)
                        else:
                            depth = np.load(f'{file_name}')
                        depth = cv2.resize(depth, (opt.image_size, opt.image_size),
                                           interpolation=cv2.INTER_CUBIC)
                        depth = torch.tensor(depth).view(1, opt.image_size, opt.image_size)
                        camera_objects[file_name[-10:-4]].append(depth)
            for dir_data in pose_dirs:
                if opt.dataset == 'airsim-mrmps-data' or opt.dataset == 'airsim-mrmps-noise-data':
                    files_path = pose_path + dir_data + '/' + camera_names[i]
                else:
                    files_path = pose_path + camera_names[i]
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
            for k, v in severity_dict.items():
                camera_objects[k].append(v)
            for k, v in camera_objects.items():
                if len(v) == 5:
                    consistent_objects.append(v)

            print(f'[Saving {all_data_path[i]} to disk]')
            torch.save(consistent_objects, all_data_path[i])

            del consistent_objects, camera_objects

        print(f'[Reload all data to get valid index]')
        consistent_camera_objects = [[] for i in range(opt.camera_num)]
        consistent_camera_id = {}
        for i in range(len(all_data_path)):
            data = torch.load(all_data_path[i])
            for j in range(len(data)):
                if not data[j][0] in consistent_camera_id:
                    consistent_camera_id[data[j][0]] = [[data[j][1], data[j][2], data[j][3], data[j][4]]]
                else:
                    consistent_camera_id[data[j][0]].append([data[j][1], data[j][2], data[j][3], data[j][4]])
        for k, v in consistent_camera_id.items():
            if len(v) == opt.camera_num:
                for i in range(opt.camera_num):
                    consistent_camera_objects[i].append([v[i][0], v[i][1], v[i][2], v[i][3]])
        for i in range(len(all_data_path)):
            torch.save(consistent_camera_objects[i], all_data_path[i])
        del consistent_camera_objects, consistent_camera_id

    print(f'[Loading all data]')
    print(all_data_path)
    for i in range(len(all_data_path)):
        data = torch.load(all_data_path[i])
        for j in range(len(data)):
            images[i].append(data[j][0])
            depths[i].append(data[j][1])
            poses[i].append(data[j][2])
            severities[i].append(data[j][3])

    return images, depths, poses, severities, real_camera_num


def load_splits_stats(images, depths, opt):
    splits_path = opt.dataset + '/splits.pth'
    if os.path.exists(splits_path):
        print(f'[loading data splits: {splits_path}]')
        splits = torch.load(splits_path)
        n_samples = splits.get('n_samples')
        train_val_indx = splits.get('train_val_indx')
        test_indx = splits.get('test_indx')
    else:
        if opt.apply_noise_idx is not None:
            print('splits.pth not found! splits for noise_data should be copied from data.')
            raise KeyError
        else:
            print('[generating data splits]')
            rgn = numpy.random.RandomState(0)
            n_samples = len(images[0])
            perm = rgn.permutation(n_samples)
            n_train_val = int(math.floor(n_samples * 0.9))
            train_val_indx = perm[0: n_train_val]
            test_indx = perm[n_train_val:]
            torch.save(dict(
                n_samples=n_samples,
                train_val_indx=train_val_indx,
                test_indx=test_indx,
            ), splits_path)
    stats_path = opt.dataset + '/data_stats.pth'
    if os.path.isfile(stats_path):
        print(f'[loading data stats: {stats_path}]')
        stats = torch.load(stats_path)
    else:
        if opt.apply_noise_idx is not None:
            print('data_stats.pth not found! data_stats for noise_data should be copied from data.')
            raise KeyError
        else:
            print('[computing image and depth stats]  camera_num=', opt.camera_num)
            assert opt.camera_num == 5
            stat_images = [[] for i in range(opt.camera_num)]
            stat_depths = [[] for i in range(opt.camera_num)]
            for i in range(opt.camera_num):
                stat_images[i] = torch.stack(images[i], dim=0)
                stat_depths[i] = torch.stack(depths[i], dim=0)
            stat_images = torch.stack(stat_images, dim=1)
            stat_depths = torch.stack(stat_depths, dim=1)
            stats = dict()
            all_images = stat_images.view(-1, 3, stat_images.size(3), stat_images.size(4))
            all_depths = stat_depths.view(-1, 1, stat_depths.size(3), stat_depths.size(4))
            # Compute mean and std for each channel
            stats['images_mean'] = torch.mean(all_images, (0, 2, 3))
            stats['images_std'] = torch.std(all_images, (0, 2, 3))
            stats['depths_mean'] = torch.mean(all_depths, (0, 2, 3))
            stats['depths_std'] = torch.std(all_depths, (0, 2, 3))
            torch.save(stats, stats_path)

    return n_samples, train_val_indx, test_indx, stats


def apply_noise_on_images(image, opt):
    noise_operation = None
    noise = random.random()
    if noise > 0.2:
        severity = random.randint(1, 5)
        if noise > 0.2 and noise <= 0.3:
            noise_operation = gaussian_noise
        elif noise > 0.3 and noise <= 0.4:
            noise_operation = shot_noise
        elif noise > 0.4 and noise <= 0.5:
            noise_operation = impulse_noise
        elif noise > 0.5 and noise <= 0.7:
            noise_operation = motion_blur
        elif noise > 0.7 and noise <= 0.9:
            noise_operation = snow
        else:
            noise_operation = jpeg_compression
    else:
        severity = 0
    image = image.convert('RGB')
    image = transforms.Resize((opt.image_size, opt.image_size))(image)
    if noise_operation is not None:
        image = noise_operation(image, severity)
    image = transforms.ToTensor()(image)
    return image, severity


def general_normalisation(objects, mean, std, name):
    if name == 'image':
        dim = 3
    else:
        dim = 1
    objects -= mean.view(dim, 1, 1)
    objects /= std.view(dim, 1, 1)
    return objects


def general_unormalisation(objects, mean, std, name, use_cuda=True):
    if name == 'image':
        dim = 3
    else:
        dim = 1
    if use_cuda:
        objects *= std.view(1, 1, dim, 1, 1).cuda()
        objects += mean.view(1, 1, dim, 1, 1).cuda()
    else:
        objects *= std.view(dim, 1, 1)
        objects += mean.view(dim, 1, 1)
    return objects
