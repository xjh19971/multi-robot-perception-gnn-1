import argparse
import cv2
import glob
import math
import numpy
import numpy as np
import os
import random
import torch
from PIL import Image
from torchvision import transforms
from imagenet_c import gaussian_noise, shot_noise, impulse_noise, motion_blur, snow, jpeg_compression

from dgl import load_graphs, graph, save_graphs
from dgl.data import DGLDataset
from dgl.convert import graph as dgl_graph
from utils import cal_relative_pose, AddPepperNoise

class MultiViewDGLDataset(DGLDataset):
    """
    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 opt,
                 url=None,
                 raw_dir='airsim-mrmps-data',
                 save_dir='airsim-mrmps-process',
                 force_reload=False,
                 verbose=False):
        self.opt = opt
        self.img_transforms = transforms.Compose([
                                    transforms.Resize((self.opt.image_size, self.opt.image_size)),
                                    transforms.ToTensor(),
                                ])

        if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
            pname = 'AirsimDGL'
        else:
            pname = 'FlightmareDGL'
        print(f"[Initialize {pname}]")
        print("save_dir: ", save_dir)
        super(MultiViewDGLDataset, self).__init__(name=pname,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose
                                        )
        


    @property
    def camera_names(self):
        return ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']
    @property
    def camera_num(self):
        return

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        print("[Process Dataset]")
        random.seed(self.opt.seed)
        numpy.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)

        if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
            image_path = self.opt.dataset + '/scene/async_rotate_fog_000_clear/'
            depth_path = self.opt.dataset + '/depth_encoded/async_rotate_fog_000_clear/'
            pose_path = self.opt.dataset + '/pose/async_rotate_fog_000_clear/'
        else:
            image_path = self.opt.dataset + '/scene/'
            depth_path = self.opt.dataset + '/depth_encoded/'
            pose_path = self.opt.dataset + '/pose/'
        all_data_path = []
        if self.opt.target == 'test':
            self.real_camera_num = len(self.camera_names)
        else:
            self.real_camera_num = self.opt.camera_num
        for i in self.opt.camera_idx:
            all_data_path.append(self.opt.dataset + '/' + self.camera_names[i] + '_all_data.pth')

        self.images = [[] for i in range(self.real_camera_num)]
        self.depths = [[] for i in range(self.real_camera_num)]
        self.poses = [[] for i in range(self.real_camera_num)]
        self.severities = [[] for i in range(self.real_camera_num)]

        if not os.path.exists(all_data_path[-1]):
            assert (self.opt.camera_num == 5)
            if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
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
                severity_dict = {}
                for dir_data in image_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        files_path = image_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = image_path + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        noise_operation = None
                        severity = -1
                        image = Image.open(f'{file_name}')
                        if self.img_transforms is not None:
                            if self.opt.apply_noise_idx is not None and i in self.opt.apply_noise_idx:
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
                                image = transforms.Resize((self.opt.image_size, self.opt.image_size))(image)
                                if noise_operation is not None:
                                    image = noise_operation(image, severity)
                                image = transforms.ToTensor()(image)
                            else:
                                image = self.img_transforms(image)
                            image = image[0:3, :, :]
                        camera_objects[file_name[-10:-4]] = [file_name[-10:-4], image]
                        severity_dict[file_name[-10:-4]] = severity
                for dir_data in depth_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        files_path = depth_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = depth_path + self.camera_names[i]
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        file_names = glob.glob(f'{files_path}/*.png')
                    else:
                        file_names = glob.glob(f'{files_path}/*.npy')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                                depth = cv2.imread(f'{file_name}')
                                depth = np.array(
                                    depth[:, :, 0] * (256 ** 3) + depth[:, :, 1] * (256 ** 2) + depth[:, :, 2] * (
                                            256 ** 1),
                                    dtype=np.uint32)
                                depth = depth.view(np.float32)
                            else:
                                depth = np.load(f'{file_name}')
                            depth = cv2.resize(depth, (self.opt.image_size, self.opt.image_size),
                                               interpolation=cv2.INTER_CUBIC)
                            depth = torch.tensor(depth).view(1, self.opt.image_size, self.opt.image_size)
                            camera_objects[file_name[-10:-4]].append(depth)
                for dir_data in pose_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
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
                for k, v in severity_dict.items():
                    camera_objects[k].append(v)
                for k, v in camera_objects.items():
                    if len(v) == 5:
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
                        consistent_camera_id[data[j][0]] = [[data[j][1], data[j][2], data[j][3], data[j][4]]]
                    else:
                        consistent_camera_id[data[j][0]].append([data[j][1], data[j][2], data[j][3], data[j][4]])
            for k, v in consistent_camera_id.items():
                if len(v) == self.opt.camera_num:
                    for i in range(self.opt.camera_num):
                        consistent_camera_objects[i].append([v[i][0], v[i][1], v[i][2], v[i][3]])
            for i in range(len(all_data_path)):
                torch.save(consistent_camera_objects[i], all_data_path[i])
            del consistent_camera_objects, consistent_camera_id

        print(f'[Loading all data]', all_data_path)
        for i in range(len(all_data_path)):
            data = torch.load(all_data_path[i])
            for j in range(len(data)):
                self.images[i].append(data[j][0])
                self.depths[i].append(data[j][1])
                self.poses[i].append(data[j][2])
                self.severities[i].append(data[j][3])

        splits_path = self.opt.dataset + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.n_samples = self.splits.get('n_samples')
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
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
             if self.opt.apply_noise_idx is not None:
                print('data_stats.pth not found! data_stats for noise_data should be copied from data.')
                raise KeyError
             else:
                print('[computing image and depth stats]  camera_num=', self.opt.camera_num )
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




        print(f'[Construct graphs]')
        x= []
        y= []
        for i in range(self.opt.camera_num):
            for j in range(self.opt.camera_num):
                if not (i==j):
                    x.append(i)
                    y.append(j)
        edge_list = (x,y)

        self.graphs = []
        for item in range(self.n_samples):
            g=graph(edge_list)
            image_set = []
            depth_set = []
            edge_set = []
            feature_set = []
            for cam in range(self.opt.camera_num):
                image_set.append(self.normalise_object(self.images[cam][item], self.stats['images_mean'], self.stats['images_std'],'image'))
                depth_set.append(self.depths[cam][item])
                feature_set.append(torch.zeros(8,8,1280))
            g.ndata['image'] = torch.stack(image_set, dim=0)
            g.ndata['depth'] = torch.stack(depth_set, dim=0)
            #g.ndata['feature'] = torch.stack(feature_set, dim=0)
            for i in range(self.opt.camera_num):
                for j in range(self.opt.camera_num):
                    if not (i==j):
                        relative_pose = cal_relative_pose(self.poses[i][item][1:].numpy(), self.poses[j][item][1:].numpy())
                        edge_set.append(torch.from_numpy(relative_pose))
            g.edata['pose'] = torch.stack(edge_set,dim=0)
            self.graphs.append(g)

    @staticmethod
    def normalise_object(objects, mean, std, name):
        if name == 'image':
            dim = 3
        else:
            dim = 1
        objects -= mean.view(dim, 1, 1)
        objects /= std.view(dim, 1, 1)
        return objects

    @staticmethod
    def unormalise_object(objects, mean, std, name, use_cuda=True):
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
    @property
    def images_mean(self):
        return self.stats['images_mean']
    @property
    def images_std(self):
        return self.stats['images_std']
    @property
    def depths_mean(self):
        return self.stats['depths_mean']
    @property
    def depths_std(self):
        return self.stats['depths_std']
    @property
    def camera_num(self):
        return self.opt.camera_num

    def __getitem__(self, idx):
        # get one example by index
        if self.opt.target == 'test':
            real_index = self.test_indx[idx]
        else:
            real_index = self.train_val_indx[idx]
        return self.graphs[real_index]

    def __len__(self):
        # number of data examples
        return len(self.test_indx) if self.opt.target == 'test' else len(self.train_val_indx)

    def save(self):
        # save processed data to directory `self.save_path`
        print('[Save graphs]')
        graph_path = os.path.join(self.save_dir, 'dgl_graph_'+str(self.camera_num)+'.bin')
        save_graphs(str(graph_path), self.graphs)

    def load(self):
        # load processed data from directory `self.save_path`
        print('[Load existed graph]')
        graphs, _ = load_graphs(os.path.join(self.save_dir, 'dgl_graph_'+str(self.camera_num)+'.bin'))
        self.graphs = graphs

        splits_path = self.save_dir + '/splits-'+str(self.camera_num)+'.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.n_samples = self.splits.get('n_samples')
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            raise NameError('splits.pth not existed')
        stats_path = self.save_dir + '/data_stats-'+str(self.camera_num)+'.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            self.stats = torch.load(stats_path)
            print(f'[Finish loading data stats]')
        else:
            raise NameError('stats_path.pth not existed')

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_dir,self.opt.dataset+'_dgl_graph_'+str(self.camera_num)+'.bin')
        if os.path.exists(graph_path):
            print('[Cached]')
            return True
        else:
            print('[Not Cached]')
            return False

class SingleViewDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']

        self.img_transforms = transforms.Compose([
            transforms.Resize((self.opt.image_size, self.opt.image_size)),
            transforms.ToTensor(),
        ])
        random.seed(self.opt.seed)
        numpy.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)

        if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
            image_path = self.opt.dataset + '/scene/async_rotate_fog_000_clear/'
            depth_path = self.opt.dataset + '/depth_encoded/async_rotate_fog_000_clear/'
            pose_path = self.opt.dataset + '/pose/async_rotate_fog_000_clear/'
        else:
            image_path = self.opt.dataset + '/scene/'
            depth_path = self.opt.dataset + '/depth_encoded/'
            pose_path = self.opt.dataset + '/pose/'
        all_data_path = []
        if self.opt.target == 'test':
            self.real_camera_num = len(self.camera_names)
        else:
            self.real_camera_num = self.opt.camera_num
        for i in self.opt.camera_idx:
            all_data_path.append(self.opt.dataset + '/' + self.camera_names[i] + '_all_data.pth')

        self.images = [[] for i in range(self.real_camera_num)]
        self.depths = [[] for i in range(self.real_camera_num)]
        self.poses = [[] for i in range(self.real_camera_num)]
        self.severities = [[] for i in range(self.real_camera_num)]

        if not os.path.exists(all_data_path[-1]):
            assert (self.opt.camera_num == 5)
            if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
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
                severity_dict = {}
                for dir_data in image_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        files_path = image_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = image_path + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        noise_operation = None
                        severity = -1
                        image = Image.open(f'{file_name}')
                        if self.img_transforms is not None:
                            if self.opt.apply_noise_idx is not None and i in self.opt.apply_noise_idx:
                                noise = random.random()
                                if noise > 0.2:
                                    severity=random.randint(1,5)
                                    if noise>0.2 and noise<=0.3:
                                        noise_operation=gaussian_noise
                                    elif noise>0.3 and noise<=0.4:
                                        noise_operation = shot_noise
                                    elif noise>0.4 and noise<=0.5:
                                        noise_operation = impulse_noise
                                    elif noise>0.5 and noise<=0.7:
                                        noise_operation = motion_blur
                                    elif noise>0.7 and noise<=0.9:
                                        noise_operation = snow
                                    else:
                                        noise_operation = jpeg_compression
                                else:
                                    severity = 0
                                image=image.convert('RGB')
                                image = transforms.Resize((self.opt.image_size, self.opt.image_size))(image)
                                if noise_operation is not None:
                                    image = noise_operation(image, severity)
                                image = transforms.ToTensor()(image)
                            else:
                                image = self.img_transforms(image)
                            image = image[0:3, :, :]
                        camera_objects[file_name[-10:-4]] = [file_name[-10:-4], image]
                        severity_dict[file_name[-10:-4]] = severity
                for dir_data in depth_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        files_path = depth_path + dir_data + '/' + self.camera_names[i]
                    else:
                        files_path = depth_path + self.camera_names[i]
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                        file_names = glob.glob(f'{files_path}/*.png')
                    else:
                        file_names = glob.glob(f'{files_path}/*.npy')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
                                depth = cv2.imread(f'{file_name}')
                                depth = np.array(
                                    depth[:, :, 0] * (256 ** 3) + depth[:, :, 1] * (256 ** 2) + depth[:, :, 2] * (
                                                256 ** 1),
                                    dtype=np.uint32)
                                depth = depth.view(np.float32)
                            else:
                                depth = np.load(f'{file_name}')
                            depth = cv2.resize(depth, (self.opt.image_size, self.opt.image_size),
                                               interpolation=cv2.INTER_CUBIC)
                            depth = torch.tensor(depth).view(1, self.opt.image_size, self.opt.image_size)
                            camera_objects[file_name[-10:-4]].append(depth)
                for dir_data in pose_dirs:
                    if self.opt.dataset == 'airsim-mrmps-data' or self.opt.dataset == 'airsim-mrmps-noise-data':
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
                for k, v in severity_dict.items():
                    camera_objects[k].append(v)
                for k, v in camera_objects.items():
                    if len(v) == 5:
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
                        consistent_camera_id[data[j][0]] = [[data[j][1], data[j][2], data[j][3], data[j][4]]]
                    else:
                        consistent_camera_id[data[j][0]].append([data[j][1], data[j][2], data[j][3], data[j][4]])
            for k, v in consistent_camera_id.items():
                if len(v) == self.opt.camera_num:
                    for i in range(self.opt.camera_num):
                        consistent_camera_objects[i].append([v[i][0], v[i][1], v[i][2], v[i][3]])
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
                self.severities[i].append(data[j][3])

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
            if self.opt.apply_noise_idx is not None:
                print('splits.pth not found! splits for noise_data should be copied from data.')
                raise KeyError
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
             if self.opt.apply_noise_idx is not None:
                print('data_stats.pth not found! data_stats for noise_data should be copied from data.')
                raise KeyError
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
        # # pdb.set_trace()
        # for i in range(self.opt.camera_num):
        #     single_data = [data[0][i, :, :, :], data[1][:, i, :, :, :].squeeze(0), data[2][:, i, :].squeeze(0)]
        #     self.generated_dataset[i][self.generated_indx[idx]] = single_data
        pass

    def store_all(self, path, model_num):
        # for i in range(len(self.camera_names)):
        #     real_path = path+self.camera_names[i]+f'_all_data_m{model_num.group(0)[-1]}.pth'
        #     print(f'[storing feature map at {real_path}]')
        #     torch.save(self.generated_dataset[i], real_path)
        pass

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
            objects *= std.view(1, 1, dim, 1, 1).cuda()
            objects += mean.view(1, 1, dim, 1, 1).cuda()
        else:
            objects *= std.view(dim, 1, 1)
            objects += mean.view(dim, 1, 1)
        return objects
