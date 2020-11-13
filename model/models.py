import torch
import torch.nn as nn
from model.backbone import resnet_wrapper
class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt

        # image encoder
        assert (opt.nfeature % 4 == 0)
        self.image_encoder = resnet_wrapper("resnet50")

        # pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.opt.npose, self.opt.nfeature_pose),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature_pose, self.opt.nfeature_pose),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature_pose, self.opt.image_size//32)
        )

    def forward(self, images, poses):
        images = images.view(-1, images.size(1)//self.opt.camera_num, self.opt.image_size, self.opt.image_size)
        poses = poses.view(-1, images.size(1)//self.opt.camera_num)
        h = self.image_encoder(images)[-1]
        h += self.pose_encoder(poses)
        return h


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt

        self.feature_maps = (self.opt.nfeature_image // 4, self.opt.nfeature_image // 2, self.opt.nfeature_image)

        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_maps[2], self.feature_maps[1], 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_maps[1], self.feature_maps[0], 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_maps[0], 3, 4, 2, 1)
        )

        # pose_decoder?

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.feature_maps[-1], self.opt.h_height, self.opt.h_width)
        pred_image = self.image_decoder(h)
        pred_image = pred_image.view(bsize, 3, self.opt.height, self.opt.width)
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
