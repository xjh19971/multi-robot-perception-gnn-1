import sys
import numpy, random, pdb, math, pickle, glob, time, os, re, argparse
import torch
from PIL import Image
from torchvision import transforms

class MRPGDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']
        #self.camera_names = ['DroneNN_main', 'DroneNP_main']
        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        image_path = self.opt.dataset + '/depth/async_rotate_fog_000_clear/'
        depth_path = self.opt.dataset + '/scene/async_rotate_fog_000_clear/'
        pose_path = self.opt.dataset + '/pose/async_rotate_fog_000_clear/'
        all_data_path = []
        for camera in self.camera_names:
            all_data_path.append(self.opt.dataset + '/' + camera + '_all_data.pth')

        self.images = [[] for i in range(len(self.camera_names))]
        self.depths = [[] for i in range(len(self.camera_names))]
        self.poses = [[] for i in range(len(self.camera_names))]

        if not os.path.exists(all_data_path[-1]):
            image_dirs = next(os.walk(image_path))[1]
            image_dirs.sort()
            depth_dirs = next(os.walk(depth_path))[1]
            depth_dirs.sort()
            pose_dirs = next(os.walk(pose_path))[1]
            pose_dirs.sort()

            for i in range(len(self.camera_names)):
                camera_objects = {}
                for dir_data in image_dirs:
                    files_path = image_path + dir_data + '/' + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        image = Image.open(f'{file_name}')
                        if self.img_transforms is not None:
                            image = self.img_transforms(image)
                        camera_objects[file_name[-10:-4]] = [file_name[-10:-4],image]
                for dir_data in depth_dirs:
                    files_path = depth_path + dir_data + '/' + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            depth = Image.open(f'{file_name}')
                            if self.img_transforms is not None:
                                depth = self.img_transforms(depth)
                            camera_objects[file_name[-10:-4]].append(depth)
                for dir_data in pose_dirs:
                    files_path = pose_path + dir_data + '/' + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.txt')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            with open(f'{file_name}', 'r') as f:
                                pose = f.readlines()
                                pose = list(map(float, pose))
                                camera_objects[file_name[-10:-4]].append(pose)

                consistent_objects = []
                for k, v in camera_objects.items():
                    if len(v) == 4:
                        consistent_objects.append(v)

                print(f'[Saving {all_data_path[i]} to disk]')
                torch.save(consistent_objects, all_data_path[i])

        print(f'[Loading all data]')
        consistent_camera_id = {}
        for i in range(len(all_data_path)):
            data = torch.load(all_data_path[i])
            for j in range(len(data)):
                if not data[j][0] in consistent_camera_id:
                    consistent_camera_id[data[j][0]] = [[data[j][1], data[j][2], data[j][3]]]
                else:
                    consistent_camera_id[data[j][0]].append([data[j][1], data[j][2], data[j][3]])
        for k, v in consistent_camera_id.items():
            if len(v) == len(self.camera_names):
                for i in range(len(self.camera_names)):
                    self.images[i].append(v[i][0])
                    self.depths[i].append(v[i][1])
                    self.poses[i].append(v[i][2])
        self.n_samples = len(self.images[0])
        print(f'[Number of samples for each camera: {self.n_samples}]')

        splits_path = self.opt.dataset + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading test data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating test data splits]')
            rgn = numpy.random.RandomState(0)
            perm = rgn.permutation(self.n_samples)
            n_train_val = int(math.floor(self.n_samples * 0.9))
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
                    all_poses.append(torch.tensor(self.poses[i][j]))
            all_poses = torch.stack(all_poses, 0)
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
        image = []
        pose = []
        depth = []
        for i in range(len(self.camera_names)):
            image.append(self.images[i][real_index])
            pose.append(torch.tensor(self.poses[i][real_index]))
            depth.append(self.depths[i][real_index])
        image = torch.stack(image, dim=0)
        pose = torch.stack(pose,dim=0)
        depth = torch.stack(depth,dim=0)
        # image = self.normalise_image(image)
        pose = self.normalise_pose(pose)
        # depth = self.normalise_image(depth)

        return image, pose, depth

    @staticmethod
    def normalise_image(images):
        return images.float().div_(255.0)

    def normalise_pose(self, pose):
        pose -= self.poses_mean.view(1, 8)
        pose /= (1e-8 + self.poses_std.view(1, 8))
        return pose

    @staticmethod
    def unormalse_image(images):
        return images.mul_(255.0).byte()

    def unnormalise_pose(self, pose):
        pose *= (1e-8 + self.poses_std.view(1, 8))
        pose += self.poses_mean.view(1, 8)
        return pose

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    a_batch = next(iter(trainloader))
