import sys
import numpy, random, pdb, math, pickle, glob, time, os, re
import torch
import cv2


class MRPGDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        self.camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']
        image_path = self.opt.dataset + '/depth/async_rotate_fog_000_clear/'
        depth_path = self.opt.dataset + '/scene/async_rotate_fog_000_clear/'
        pose_path = self.opt.dataset + '/pose/async_rotate_fog_000_clear/'
        all_data_path = self.opt.dataset + '/all_data.pth'
        if not os.path.exists(all_data_path):
            image_dirs = next(os.walk(image_path))[1]
            image_dirs.sort()
            depth_dirs = next(os.walk(depth_path))[1]
            depth_dirs.sort()
            pose_dirs = next(os.walk(pose_path))[1]
            pose_dirs.sort()

            self.images = []
            self.depths = []
            self.poses = []

            for camera in self.camera_names:
                camera_images = []
                camera_depths = []
                camera_poses = []
                ids = []
                for dir_data in image_dirs:
                    files_path = image_path + dir_data + '/' + camera
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        image = cv2.imread(f'{file_name}')
                        camera_images.append({file_name:image})
                        ids.append(file_name[-10:-4])

                for dir_data in depth_dirs:
                    files_path = depth_path + dir_data + '/' + camera
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in ids:
                            depth = cv2.imread(f'{file_name}')
                            camera_depths.append({file_name:depth})

                for dir_data in pose_dirs:
                    files_path = pose_path + dir_data + '/' + camera
                    file_names = glob.glob(f'{files_path}/*.txt')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in ids:
                            with open(f'{file_name}', 'r') as f:
                                pose = f.readlines()
                                camera_poses.append({file_name:pose})

                self.images.append(camera_images)
                self.depths.append(camera_depths)
                self.poses.append(camera_poses)

            print(f'Saving {all_data_path} to disk')
            torch.save({
                'images': self.images,
                'depths': self.depths,
                'poses': self.poses
            }, all_data_path)
        else:
            print(f'[loading all data: {all_data_path}]')
            data = torch.load(all_data_path)
            self.images = data.get('images')
            self.depths = data.get('depths')
            self.poses = data.get('poses')

        self.n_samples = len(self.images[0])
        print(f'Number of samples for each camera: {self.n_episodes}')

        splits_path = self.opt.dataset + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading test data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating test data splits]')
            rgn = numpy.random.RandomState(0)
            perm = rgn.permutation(self.n_episodes)
            n_train_val = int(math.floor(self.n_episodes * 0.9))
            self.train_val_indx = perm[0: n_train_val]
            self.test_indx = perm[n_train_val:]
            torch.save(dict(
                train_val_indx=self.train_val_indx,
                test_indx=self.test_indx,
            ), splits_path)

        stats_path = self.opt.dataset + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            stats = torch.load(stats_path)
            self.poses_mean = stats.get('poses_mean')
            self.poses_std = stats.get('poses_std')
        else:
            print('[computing action stats]')
            all_poses = []
            for i in range(len(self.camera_names)):
                for j in range(len(self.poses[i])):
                    all_poses.append(self.poses[i][j])
            all_poses = torch.cat(all_poses, 0)
            self.poses_mean = torch.mean(all_poses, 0)
            self.poses_std = torch.std(all_poses, 0)

            torch.save({'poses_mean': self.poses_mean,
                        'poses_std': self.poses_std}
                       , stats_path)

    def __len__(self):
        return len(self.test_indx) if self.opt.target == 'test' else len(self.train_val_indx)

    def __getitem__(self, index):
        if self.opt.target == 'test':
            real_index = self.test_indx[index]
        else:
            real_index = self.train_val_indx[index]
        image = self.images[real_index]
        pose = self.poses[real_index]
        depth = self.depths[real_index]
        pose = self.normalise_pose(pose)
        return image, pose, depth

    @staticmethod
    def normalise_image(images):
        return images.float().div_(255.0)

    def normalise_pose(self, pose):
        pose -= self.poses_mean.view(1, 8)
        pose /= (1e-8 + self.poses_std.view(1, 8))
        return pose
