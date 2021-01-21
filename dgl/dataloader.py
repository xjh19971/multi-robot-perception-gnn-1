import os
import random
import re

import numpy
import torch
from dataloader_utils import load_raw_data, load_splits_stats, general_normalisation, general_unormalisation
from dgl.data import DGLDataset
from torchvision import transforms
from utils import cal_relative_pose

from dgl import load_graphs, graph, save_graphs


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
        self.img_transforms = transforms.Compose([transforms.Resize((self.opt.image_size, self.opt.image_size)),
                                                  transforms.ToTensor()])

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

        self.images, self.depths, self.poses, self.severities = load_raw_data(self.opt, self.camera_names,
                                                                              self.img_transforms)

        self.n_samples, self.train_val_indx, self.test_indx, self.stats = load_splits_stats(self.images, self.depths,
                                                                                            self.opt)

        print(f'[Number of samples for each camera: {self.n_samples}]')
        print(f'[Construct graphs]')
        x = []
        y = []
        for i in range(self.opt.camera_num):
            for j in range(self.opt.camera_num):
                if not (i == j):
                    x.append(i)
                    y.append(j)
        edge_list = (x, y)

        self.graphs = []
        for item in range(self.n_samples):
            g = graph(edge_list)
            image_set = []
            depth_set = []
            edge_set = []
            feature_set = []
            for cam in range(self.opt.camera_num):
                image_set.append(
                    self.normalise_object(self.images[cam][item], self.stats['images_mean'], self.stats['images_std'],
                                          'image'))
                depth_set.append(self.depths[cam][item])
                feature_set.append(torch.zeros(8, 8, 1280))
            g.ndata['image'] = torch.stack(image_set, dim=0).float()
            g.ndata['depth'] = torch.stack(depth_set, dim=0)
            # g.ndata['feature'] = torch.stack(feature_set, dim=0)
            for i in range(self.opt.camera_num):
                for j in range(self.opt.camera_num):
                    if not (i == j):
                        relative_pose = cal_relative_pose(self.poses[i][item][1:].numpy(),
                                                          self.poses[j][item][1:].numpy())
                        edge_set.append(torch.from_numpy(relative_pose))
            g.edata['pose'] = torch.stack(edge_set, dim=0).float()
            self.graphs.append(g)

    @staticmethod
    def normalise_object(objects, mean, std, name):
        return general_normalisation(objects, mean, std, name)

    @staticmethod
    def unormalise_object(objects, mean, std, name, use_cuda=True):
        return general_unormalisation(objects, mean, std, name, use_cuda)

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
        graph_path = os.path.join(self.save_dir, 'dgl_graph_' + str(self.camera_num) + '.bin')
        save_graphs(str(graph_path), self.graphs)

    def load(self):
        # load processed data from directory `self.save_path`
        print('[Load existed graph]')
        graphs, _ = load_graphs(os.path.join(self.save_dir, 'dgl_graph_' + str(self.camera_num) + '.bin'))
        self.graphs = graphs

        splits_path = self.opt.dataset + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.n_samples = self.splits.get('n_samples')
            self.train_val_indx = self.splits.get('train_val_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            raise NameError('splits.pth not existed')
        stats_path = self.opt.dataset + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            self.stats = torch.load(stats_path)
            print(f'[Finish loading data stats]')
        else:
            raise NameError('stats_path.pth not existed')

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_dir, 'dgl_graph_' + str(self.camera_num) + '.bin')
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

        self.images, self.depths, self.poses, self.severities = load_raw_data(self.opt, self.camera_names,
                                                                              self.img_transforms)

        self.n_samples, self.train_val_indx, self.test_indx, self.stats = load_splits_stats(self.images, self.depths,
                                                                                            self.opt)

        print(f'[Number of samples for each camera: {self.n_samples}]')

    def __len__(self):
        if self.opt.target == 'test':
            return len(self.test_indx)
        else:
            return len(self.train_val_indx)

    def __getitem__(self, index):
        if self.opt.target == 'test':
            real_index = self.test_indx[index]
        else:
            real_index = self.train_val_indx[index]
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
        image = image.float()
        pose = pose.float()
        depth = depth.float()
        return image, pose, depth

    @staticmethod
    def normalise_object(objects, mean, std, name):
        return general_normalisation(objects, mean, std, name)

    @staticmethod
    def unormalise_object(objects, mean, std, name, use_cuda=True):
        return general_unormalisation(objects, mean, std, name, use_cuda)


def generate_dataset(opt):
    dataset = None

    if opt.dataset == "airsim":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "airsim-noise":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "airsim-dgl":
        opt.dataset = "airsim-mrmps-data"
        print(f'[Loading airsim MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="airsim-mrmps-data", save_dir="airsim-mrmps-process")
        print(dataset[0])
    elif opt.dataset == "airsim-noise-dgl":
        opt.dataset = "airsim-mrmps-noise-data"
        print(f'[Loading airsim noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="airsim-mrmps-noise-data", save_dir="airsim-mrmps-noise-process")
        print(dataset[0])

    if opt.dataset == "warehouse":
        print(f'[Loading warehouse SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "warehouse-noise":
        opt.dataset = "warehouse-noise-" + str(len(opt.apply_noise_idx))
        print(f'[Loading warehouse noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "warehouse-dgl":
        opt.dataset = "warehouse"
        print(f'[Loading warehouse MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="warehouse", save_dir="warehouse-process")
    elif opt.dataset == "warehouse-noise-dgl":
        opt.dataset = "warehouse-noise-" + str(len(opt.apply_noise_idx))
        print(f'[Loading warehouse noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="warehouse-noise-" + str(len(opt.apply_noise_idx)),
                                      save_dir="warehouse-noise-" + str(len(opt.apply_noise_idx)) + "-process")

    if opt.dataset == "industrial":
        opt.dataset = "industrial"
        print(f'[Loading industrial SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "industrial-noise":
        opt.dataset = "industrial-noise-" + str(len(opt.apply_noise_idx))
        print(f'[Loading industrial noise SingleViewDataset]')
        dataset = SingleViewDataset(opt)
    elif opt.dataset == "industrial-dgl":
        opt.dataset = "industrial"
        print(f'[Loading industrial MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="industrial", save_dir="industrial-process")
    elif opt.dataset == "industrial-noise-dgl":
        opt.dataset = "industrial-noise-" + str(len(opt.apply_noise_idx))
        print(f'[Loading industrial noise MultiViewDGLDataset]')
        dataset = MultiViewDGLDataset(opt, raw_dir="industrial-noise-" + str(len(opt.apply_noise_idx)),
                                      save_dir="industrial-noise-" + str(len(opt.apply_noise_idx)) + "-process")
    return dataset
