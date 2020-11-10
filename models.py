import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        self.pose_dim = 4
        # image encoder
        assert (opt.nfeature % 4 == 0)
        self.feature_maps = (opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, self.feature_maps[0], 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
        )

        # pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.opt.npose, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature)
        )

    def forward(self, images, poses):
        bsize = images.size(0)
        h = self.image_encoder(images.view(bsize, 3, self.opt.height, self.opt.width))
        h += self.pose_encoder(poses.view(bsize, self.pose_dim))
        return h


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt

        self.feature_maps = (self.opt.nfeature // 4, self.opt.nfeature // 2, self.opt.nfeature)

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
