import torch
import torch.nn as nn
from model.blocks import TransBlock
import torchvision.models as models
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
        images = images.view(-1, images.size(1)*images.size(2)//self.opt.camera_num, self.opt.image_size, self.opt.image_size)
        h = self.image_encoder(images)
        h = h.view(h.size(0), -1)
        return h


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        self.nfeature_image = 1280
        self.nfeature_array = [self.nfeature_image, self.nfeature_image//4, self.nfeature_image//8,
                               self.nfeature_image//16]
        self.image_decoder = nn.Sequential(
            TransBlock(self.nfeature_array[0], self.nfeature_array[0], 1),
            TransBlock(self.nfeature_array[0], self.nfeature_array[1], 2),
            TransBlock(self.nfeature_array[1], self.nfeature_array[1], 1),
            TransBlock(self.nfeature_array[1], self.nfeature_array[2], 2),
            TransBlock(self.nfeature_array[2], self.nfeature_array[2], 1),
            TransBlock(self.nfeature_array[2], self.nfeature_array[3], 2),
            TransBlock(self.nfeature_array[3], 3, 1),
            nn.Upsample(scale_factor=(4, 4)),
        )

    def forward(self, h):
        h = h.view(h.size(0), self.nfeature_image, self.opt.image_size//32, self.opt.image_size//32)
        pred_image = self.image_decoder(h)
        pred_image = pred_image.view(-1, self.opt.camera_num, 3, self.opt.image_size, self.opt.image_size)
        return pred_image


class single_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt)

    def forward(self, image, pose):
        h = self.encoder(image)
        pred_image = self.decoder(h)
        return pred_image
