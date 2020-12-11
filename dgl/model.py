import torch
import torch.nn as nn
from torchvision.models import models
from blocks import TransBlock

from dgl.nn.pytorch import GraphConv
class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        if self.opt.pretrained:
            pretrained_model = models.mobilenet_v2(pretrained=True)
        else:
            pretrained_model = models.mobilenet_v2()
        pretrained_model = pretrained_model.features
        self.image_encoder = pretrained_model

    def forward(self, images):
        # (3,256,256) -> (1024, 8, 8)  1/32 of original size
        images = images.view(-1, 3, self.opt.image_size, self.opt.image_size)
        h = self.image_encoder(images)
        return h


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        self.nfeature_image = 1280
        self.nfeature_array = [self.nfeature_image, self.nfeature_image // 4, self.nfeature_image // 8,
                               self.nfeature_image // 16]
        self.image_decoder = nn.Sequential(
            TransBlock(self.nfeature_array[0], self.nfeature_array[0], 1),
            TransBlock(self.nfeature_array[0], self.nfeature_array[1], 2),
            TransBlock(self.nfeature_array[1], self.nfeature_array[1], 1),
            TransBlock(self.nfeature_array[1], self.nfeature_array[2], 2),
            TransBlock(self.nfeature_array[2], self.nfeature_array[2], 1),
            TransBlock(self.nfeature_array[2], self.nfeature_array[3], 2),
            TransBlock(self.nfeature_array[3], 1, 1),
            nn.Upsample(scale_factor=(4, 4), mode="bicubic"),
            nn.ReLU()
        )

    def forward(self, h):
        pred_image = self.image_decoder(h)
        pred_image = pred_image.view(-1, self.opt.camera_num, 1, self.opt.image_size, self.opt.image_size)
        return pred_image

class single_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt)

    def forward(self, image, pose, extract_feature):
        h = self.encoder(image)
        if extract_feature:
            return h
        pred_image = self.decoder(h)
        return pred_image


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
