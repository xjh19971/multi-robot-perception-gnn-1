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


from dgl import load_graphs
from dgl.data import DGLDataset
from dgl.convert import graph as dgl_graph
from utils import cal_relative_pose

class AirsimMapDGLDataset(DGLDataset):
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
                 raw_dir='/media/data/dataset/airsim-map/airsim-mrmps-data',
                 save_dir='/media/data/dataset/airsim-map/airsim-mrmps-process',
                 force_reload=False,
                 verbose=False):
        self.opt = opt
        self.cached = False
        self.img_transforms = transforms.Compose([
                                    transforms.Resize((self.opt.image_size, self.opt.image_size)),
                                    transforms.ToTensor(),
                                ])
        super(AirsimMapDGLDataset, self).__init__(name='AirsimMapDGL',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose,                                
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
        print(self.cached)
        if self.cached:
            self.load()
            return
        # process raw data to graphs, labels, splitting masks
        print("[Process Dataset]")
        random.seed(self.opt.seed)
        numpy.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        image_path = self.raw_dir + '/scene/async_rotate_fog_000_clear/'
        depth_path = self.raw_dir + '/depth_encoded/async_rotate_fog_000_clear/'
        pose_path = self.raw_dir + '/pose/async_rotate_fog_000_clear/'
        all_data_path = []
        for i in range(self.opt.camera_num):
            all_data_path.append(self.save_dir + '/' + self.camera_names[i] + '_all_data.pth')
        
        self.images = [[] for i in range(self.opt.camera_num)]
        self.depths = [[] for i in range(self.opt.camera_num)]
        self.poses = [[] for i in range(self.opt.camera_num)]

        if not os.path.exists(all_data_path[-1]):
            assert(self.opt.camera_num==5)
            image_dirs = next(os.walk(image_path))[1]
            image_dirs.sort()
            depth_dirs = next(os.walk(depth_path))[1]
            depth_dirs.sort()
            pose_dirs = next(os.walk(pose_path))[1]
            pose_dirs.sort()

            for i in range(self.opt.camera_num):
                camera_objects = {}
                for dir_data in image_dirs:
                    files_path = image_path + dir_data + '/' + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        image = Image.open(f'{file_name}')
                        if self.img_transforms is not None:
                            image = self.img_transforms(image)
                            image = image[0:3, :, :]
                        camera_objects[file_name[-10:-4]] = [file_name[-10:-4], image]
                for dir_data in depth_dirs:
                    files_path = depth_path + dir_data + '/' + self.camera_names[i]
                    file_names = glob.glob(f'{files_path}/*.png')
                    file_names.sort()
                    for file_name in file_names:
                        if file_name[-10:-4] in camera_objects:
                            depth = cv2.imread(f'{file_name}')
                            depth = np.array(
                                depth[:, :, 0] * (256 ** 3) + depth[:, :, 1] * (256 ** 2) + depth[:, :, 2] * (256 ** 1),
                                dtype=np.uint32)
                            depth = depth.view(np.float32)
                            depth = cv2.resize(depth, (self.opt.image_size, self.opt.image_size),
                                               interpolation=cv2.INTER_CUBIC)
                            depth = torch.tensor(depth).view(1, self.opt.image_size, self.opt.image_size)
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
                        consistent_camera_objects[i].append([v[i][0],v[i][1],v[i][2]])
            for i in range(len(all_data_path)):
                torch.save(consistent_camera_objects[i], all_data_path[i])
            del consistent_camera_objects, consistent_camera_id

        print(f'[Loading all data]')
        for i in range(len(all_data_path)):
            data = torch.load(all_data_path[i])
            for j in range(len(data)):
                self.images[i].append(data[i][0])
                self.depths[i].append(data[i][1])
                self.poses[i].append(data[i][2])

        splits_path = self.save_dir + '/splits.pth'
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
            g=dgl.graph(edge_list)
            image_set = []
            depth_set = []
            edge_set = []
            feature_set = []
            for cam in range(self.opt.camera_num):
                image_set.append(self.images[cam][item])
                depth_set.append(self.depths[cam][item])
                feature_set.append(torch.zeros(8,8,1280))
            g.ndata['image'] = torch.stack(image_set, dim=0)
            g.ndata['depth'] = torch.stack(depth_set, dim=0)
            g.ndata['feature'] = torch.stack(feature_set, dim=0)
            for i in range(self.opt.camera_num):
                for j in range(self.opt.camera_num):
                    if not (i==j):
                        relative_pose = cal_relative_pose(self.poses[i][item][1:].numpy(), self.poses[j][item][1:].numpy())
                        edge_set.append(torch.from_numpy(relative_pose))
            g.edata['pose'] = torch.stack(edge_set,dim=0)
            self.graphs.append(g)
        

    @property
    def camera_num(self):
        return self.opt.camera_num

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx]

    def __len__(self):
        # number of data examples
        return self.n_samples

    def save(self):
        if self.cached:
            return
        # save processed data to directory `self.save_path`
        print('[Save graphs]')
        graph_path = os.path.join(self.save_dir, 'dgl_graph_'+str(self.camera_num)+'.bin')
        dgl.save_graphs(str(graph_path), self.graphs)

    def load(self):
        # load processed data from directory `self.save_path`
        print('[Load existed graph]')
        graphs, _ = load_graphs(os.path.join(self.save_dir, 'dgl_graph_'+str(self.camera_num)+'.bin'))
        self.graphs = graphs
        
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_dir,'dgl_graph_'+str(self.camera_num)+'.bin')
        print(graph_path)
        if os.path.exists(graph_path):
            print('[Cached]')
            self.cached = True
            return True
        else:
            print('[Not Cached]')
            return False
        