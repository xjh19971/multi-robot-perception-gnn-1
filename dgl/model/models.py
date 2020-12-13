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
    def __init__(self, opt, nfeature_image):
        super(decoder, self).__init__()
        self.opt = opt
        self.nfeature_image = nfeature_image
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



    def forward(self, h):
        pred_image = self.image_decoder(h)
        pred_image = pred_image.view(-1, self.opt.camera_num, 1, self.opt.image_size, self.opt.image_size)
        return pred_image

class single_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt, 1280)

    def forward(self, image, pose, extract_feature):
        h = self.encoder(image)
        if extract_feature:
            return h
        pred_image = self.decoder(h)
        return pred_image

class multi_view_model(nn.Module):
    def __init__(self, opt):
        super(single_view_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.decoder = decoder(self.opt, 1280)

    def forward(self, image, pose, extract_feature):
        h = self.encoder(image)
        if extract_feature:
            return h
        pred_image = self.decoder(h)
        return pred_image


      
class edge_encoder(nn.Module):
    def __init__(self,layers_dim = [1280, 1280]):
        super(edge_encoder,self).__init__()
        self.layers_dim = layers_dim
        self.layers = nn.Sequential(
            nn.Linear(9,self.layers_dim[0]),
            nn.ReLU(),
            nn.Linear(layers_dim[0], layers_dim[1]* 2),
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
        self.gcn = GCN(self.opt)
        self.decoder = decoder(self.opt, 1280*2)

    def forward(self, g):
        with g.local_scope():
            image = g.ndata['image'] 
            image = image.view(-1, self.opt.camera_num, 3, self.opt.image_size, self.opt.image_size)
            h = self.encoder(image)
            h = h.view(-1, h.size()[-3], h.size()[-2], h.size()[-1])
            g.ndata['image'] = h
            g_h = self.gcn(g)
            h = torch.cat((h,g_h),dim=1)
            #print(h.size()) # (batch, 2560, sz, sz)
            pred_image = self.decoder(h)
            return pred_image

def node_udf(nodes):
    #print('message size: ', nodes.mailbox['m'].sum(1).size())
    #print('feature size: ', torch.cat((nodes.data['image'],nodes.mailbox['m'].sum(1)),dim=1).size())
    return {'images':nodes.mailbox['m'].mean(1)}
    #return fn.mean('m', 'images')

def edge_udf(edges):
    #print((edges.data['pose_gamma']*edges.src['image'] + edges.data['pose_beta']).size())
    return {'m': edges.data['pose_gamma']*edges.src['image'] + edges.data['pose_beta']}

class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN,self).__init__()
        self.opt = opt
        self.edge_encoder = edge_encoder()

    def forward(self,g):
        with g.local_scope():
            #g.edata['gamma'], g.edata['beta'] = self.edge_encoder(g.edata['pose'])
            g.edata['pose_gamma'], g.edata['pose_beta'] = self.edge_encoder(g.edata['pose'])
            #print('encoded edge size: ', g.edata['pose_gamma'].size())
            #g.srcdata.update({'out_src': g.ndata['image']*g.e})
            g.update_all(edge_udf,node_udf)
            return g.ndata['images']
            

