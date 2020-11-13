import torch
import torch.nn as nn
from model.backbone import resnet_wrapper
from model.blocks import TransBlock
class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        if self.opt.encoder_name=='resnet18' or self.opt.encoder_name == 'resnet34':
            self.nfeature_image = 512
        else:
            self.nfeature_image = 512*4
        # image encoder
        assert (opt.nfeature % 4 == 0)
        self.image_encoder = resnet_wrapper(self.opt.encoder_name)

        # pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.opt.npose, self.opt.nfeature_pose),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.opt.nfeature_pose, self.opt.nfeature_pose),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.opt.nfeature_pose, self.opt.image_size//32*self.opt.image_size//32*self.nfeature_image)
        )

    def forward(self, images, poses):
        images = images.view(-1, images.size(1)//self.opt.camera_num, self.opt.image_size, self.opt.image_size)
        poses = poses.view(-1, images.size(1)//self.opt.camera_num)
        h = self.image_encoder(images)[-1]
        h = h.view(h.size(0), -1)
        h += self.pose_encoder(poses)
        return h


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        if self.opt.encoder_name=='resnet18' or self.opt.encoder_name == 'resnet34':
            self.nfeature_image = 512
        else:
            self.nfeature_image = 512*4

        self.feature_maps = (self.nfeature_image//32, self.nfeature_image//16,
                             self.nfeature_image // 4, self.nfeature_image // 2, self.nfeature_image)

        self.image_decoder = nn.Sequential(
            TransBlock(self.nfeature_image // 32, self.nfeature_image // 32, 1),
            TransBlock(self.nfeature_image // 32, self.nfeature_image // 16, 2),
            TransBlock(self.nfeature_image // 16, self.nfeature_image // 16, 1),
            TransBlock(self.nfeature_image // 16, self.nfeature_image // 8, 2),
            TransBlock(self.nfeature_image // 8, self.nfeature_image // 8, 1),
            TransBlock(self.nfeature_image // 8, self.nfeature_image // 4, 2),
            TransBlock(self.nfeature_image // 4, self.nfeature_image // 4, 1),
            TransBlock(self.nfeature_image // 4, self.nfeature_image // 1, 4),
        )

        # pose_decoder?

    def forward(self, h):
        h = h.view(h.size(0), self.nfeature_image//32, self.opt.h_height, self.opt.h_width)
        pred_image = self.image_decoder(h)
        pred_image = pred_image.view(-1, pred_image.size(1)*self.opt.camera_num, self.opt.image_size, self.opt.image_size)
        return pred_image


class single_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt)

    def forward(self, image, pose):
        h = self.encoder(image, pose)
        pred_image = self.decoder(h)
        return pred_image
