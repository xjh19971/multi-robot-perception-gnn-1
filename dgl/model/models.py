import torch
import torch.nn as nn
from torchvision import models

from .blocks import TransBlock

from dgl.nn.pytorch import GraphConv
import dgl.function as fn
import dgl.ops as F

class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        feature_model = None
        if self.opt.backbone == 'mobilenetv2':
            feature_model = nn.ModuleList([nn.ModuleList() for i in range(5)])
            feature_array=[[0,1],[2,3],[4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18]]
            pretrained_model = models.mobilenet_v2(pretrained=self.opt.pretrained)
            for i in range(len(feature_array)):
                for j in range(len(feature_array[i])):
                    feature_model[i].append(pretrained_model.features[feature_array[i][j]])
        elif self.opt.backbone == 'resnet50':
            assert self.opt.skip_level == False
            feature_model = nn.ModuleList([nn.ModuleList() for i in range(5)])
            pretrained_model = models.resnet50(pretrained=self.opt.pretrained)
            pretrained_model.fc = nn.Sequential()
            pretrained_model.avgpool = nn.Sequential()
            feature_model[0].append(pretrained_model)
        elif self.opt.backbone == 'resnet18':
            assert self.opt.skip_level == False
            feature_model = nn.ModuleList([nn.ModuleList() for i in range(5)])
            pretrained_model = models.resnet18(pretrained=self.opt.pretrained)
            pretrained_model.fc = nn.Sequential()
            pretrained_model.avgpool = nn.Sequential()
            feature_model[0].append(pretrained_model)
        assert feature_model is not None
        self.image_encoder = feature_model

    def forward(self, images):
        # (3,256,256) -> (1024, 8, 8)  1/32 of original size
        images = images.view(-1, 3, self.opt.image_size, self.opt.image_size)
        h = []
        for i in range(len(self.image_encoder)):
            for layer in self.image_encoder[i]:
                images = layer(images)
            if self.opt.backbone == 'resnet18' or self.opt.backbone == 'resnet50':
                images = images.view(images.size(0), -1, self.opt.image_size//32, self.opt.image_size//32)
            h.append(images)
        return h


class decoder(nn.Module):
    def __init__(self, opt, nfeature_image):
        super(decoder, self).__init__()
        self.opt = opt
        self.nfeature_image = nfeature_image
        self.nfeature_array = [self.nfeature_image, self.nfeature_image // 2, self.nfeature_image // 4,
                               self.nfeature_image // 8, self.nfeature_image // 16, self.nfeature_image // 32]
        self.image_decoder = nn.ModuleList([
            TransBlock(self.nfeature_array[0], self.nfeature_array[1], 2, kernel_size=5),
            TransBlock(self.nfeature_array[1], self.nfeature_array[2], 2, kernel_size=5),
            TransBlock(self.nfeature_array[2], self.nfeature_array[3], 2, kernel_size=5),
            TransBlock(self.nfeature_array[3], self.nfeature_array[4], 2, kernel_size=5),
            TransBlock(self.nfeature_array[4], self.nfeature_array[5], 2, kernel_size=5),
            nn.Conv2d(self.nfeature_array[5], self.opt.output_dim, kernel_size=1)]
        )
        if self.opt.skip_level:
            input_feature_array = [96, 32, 24, 16]
            self.add_conv = nn.ModuleList([
                nn.Conv2d(input_feature_array[0], self.nfeature_array[1], kernel_size=1),
                nn.Conv2d(input_feature_array[1], self.nfeature_array[2], kernel_size=1),
                nn.Conv2d(input_feature_array[2], self.nfeature_array[3], kernel_size=1),
                nn.Conv2d(input_feature_array[3], self.nfeature_array[4], kernel_size=1)]
            )

    def forward(self, h, h_list=None):
        for i in range(len(self.image_decoder)):
            h = self.image_decoder[i](h)
            if i <= 3 and h_list is not None and self.opt.skip_level:
                h = h + self.add_conv[i](h_list[-i-2])
        pred_image = h.view(-1, self.opt.camera_num, self.opt.output_dim, self.opt.image_size, self.opt.image_size)
        return pred_image


class single_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        if self.opt.task == 'depthseg':
            temp_output_dim = self.opt.output_dim
            self.opt.output_dim = 1
            self.depth_decoder = decoder(self.opt, opt.feature_dim)
            self.opt.output_dim = temp_output_dim
            self.seg_decoder = decoder(self.opt, opt.feature_dim)
        else:
            self.decoder = decoder(self.opt, opt.feature_dim)

    def forward(self, image):
        if self.opt.skip_level:
            h_list = self.encoder(image)
            h = h_list[-1]
            if self.opt.task == 'depthseg':
                pred_depth = self.depth_decoder(h, h_list)
                pred_seg = self.seg_decoder(h, h_list)
                return pred_depth, pred_seg
            else:
                pred_image = self.decoder(h, h_list)
                return pred_image
        else:
            h_list = self.encoder(image)
            if self.opt.task == 'depthseg':
                pred_depth = self.depth_decoder(h_list[-1])
                pred_seg = self.seg_decoder(h_list[-1])
                return pred_depth, pred_seg
            else:
                pred_image = self.decoder(h_list[-1])
                return pred_image

class multi_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt, opt.feature_dim)

    def forward(self, image):
        if self.opt.skip_level:
            h_list = self.encoder(image)
            h = h_list[-1]
            pred_image = self.decoder(h, h_list)
        else:
            h = self.encoder(image)
            pred_image = self.decoder(h)
        return pred_image

## multi_view_dgl_mean
# Film edge -> gamma and beta channel-wise
# m_ij = gamma * F_i + beta
# F' = [F_i; mean (m_ij)]
# decoder feature_dim*2
class edge_encoder(nn.Module):
    def __init__(self, layers_dim):
        super(edge_encoder,self).__init__()
        self.layers_dim = layers_dim
        self.layers = nn.Sequential(
            nn.Linear(9,self.layers_dim[0]),
            nn.ReLU(),
            nn.Linear(layers_dim[0], layers_dim[1] * 2),
            nn.Sigmoid()
        )
    def forward(self, edge):
        edge = self.layers(edge.float())
        edge = edge.view(-1,self.layers_dim[1], 2) # (batch, 1280, 2)
        return edge[:,:,0].unsqueeze(-1).unsqueeze(-1), edge[:,:,1].unsqueeze(-1).unsqueeze(-1)

class multi_view_dgl_model(nn.Module):
    def __init__(self, opt):
        super(multi_view_dgl_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.gcn1 = GCN(self.opt)
        if self.opt.compress_gcn:
            self.conv1 = nn.Conv2d(opt.feature_dim * 2, opt.feature_dim, kernel_size=1)
            self.decoder = decoder(self.opt, opt.feature_dim)
        else:
            self.decoder = decoder(self.opt, opt.feature_dim * 2)
        if self.opt.multi_gcn:
            assert self.opt.compress_gcn
            self.gcn2 = GCN(self.opt)
            self.conv2 = nn.Conv2d(opt.feature_dim * 2, opt.feature_dim, kernel_size=1)

    def forward(self, g):
        with g.local_scope():
            image = g.ndata['image'] 
            image = image.view(-1, self.opt.camera_num, 3, self.opt.image_size, self.opt.image_size)
            h_list = self.encoder(image)
            h = h_list[-1]
            h = h.view(-1, h.size()[-3], h.size()[-2], h.size()[-1])
            g.ndata['image'] = h
            g_h = self.gcn1(g)
            h = torch.cat((h,g_h),dim=1)
            if self.opt.compress_gcn:
                h = self.conv1(h)
            if self.opt.multi_gcn:
                g.ndata['image'] = h
                g_h = self.gcn2(g)
                h = torch.cat((h,g_h),dim=1)
                h = self.conv2(h)
            if self.opt.skip_level:
                if self.opt.task == 'depthseg':
                    pred_depth = self.depth_decoder(h, h_list)
                    pred_seg = self.seg_decoder(h, h_list)
                    return pred_depth, pred_seg
                else:
                    pred_image = self.decoder(h, h_list)
                    return pred_image
            else:
                if self.opt.task == 'depthseg':
                    pred_depth = self.depth_decoder(h)
                    pred_seg = self.seg_decoder(h)
                    return pred_depth, pred_seg
                else:
                    pred_image = self.decoder(h)
                    return pred_image

def node_udf(nodes):
    return {'images':nodes.mailbox['m'].mean(1)}

def edge_udf(edges):
    return {'m': edges.data['pose_gamma']*edges.src['image'] + edges.data['pose_beta']}

class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN,self).__init__()
        self.opt = opt
        self.edge_encoder = edge_encoder(layers_dim = [opt.feature_dim,opt.feature_dim])

    def forward(self,g):
        with g.local_scope():
            # multi_view_dgl_mean
            g.edata['pose_gamma'], g.edata['pose_beta'] = self.edge_encoder(g.edata['pose'])
            g.update_all(edge_udf,node_udf)
            # multi_view_dgl_mean_wofilm
            #g.update_all(fn.copy_u('image','m'),node_udf)
            return g.ndata['image']
            

