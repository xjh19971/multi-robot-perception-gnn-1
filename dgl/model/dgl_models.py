## multi_view_dgl_add
# Film edge -> gamma and beta channel-wise
# m_ij = gamma * F_i + beta
# F' = F_i+ mean (m_ij)
# decoder 1280*2
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
        self.decoder = decoder(self.opt, 1280)

    def forward(self, g):
        with g.local_scope():
            image = g.ndata['image'] 
            image = image.view(-1, self.opt.camera_num, 3, self.opt.image_size, self.opt.image_size)
            h = self.encoder(image)
            h = h.view(-1, h.size()[-3], h.size()[-2], h.size()[-1])
            g.ndata['image'] = h
            g_h = self.gcn(g)
            h = g_h+ h 
            #print(h.size()) # (batch, 2560, sz, sz)
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
        self.edge_encoder = edge_encoder()

    def forward(self,g):
        with g.local_scope():
            # multi_view_dgl_mean
            g.edata['pose_gamma'], g.edata['pose_beta'] = self.edge_encoder(g.edata['pose'])
            g.update_all(edge_udf,node_udf)
            # multi_view_dgl_mean_wofilm
            #g.update_all(fn.copy_u('image','m'),node_udf)
            return g.ndata['images']
            




## multi_view_dgl_mean_wofilm
# Film edge -> gamma and beta channel-wise
# m_ij = F_i
# F' = [F_i; mean (m_ij)]
# decoder 1280*2
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
    return {'images':nodes.mailbox['m'].mean(1)}

def edge_udf(edges):
    return {'m': edges.data['pose_gamma']*edges.src['image'] + edges.data['pose_beta']}

class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN,self).__init__()
        self.opt = opt
        self.edge_encoder = edge_encoder()

    def forward(self,g):
        with g.local_scope():
            # multi_view_dgl_mean
            #g.edata['pose_gamma'], g.edata['pose_beta'] = self.edge_encoder(g.edata['pose'])
            #g.update_all(edge_udf,node_udf)
            # multi_view_dgl_mean_wofilm
            g.update_all(fn.copy_u('image','m'),node_udf)
            return g.ndata['images']




## multi_view_dgl_mean
# Film edge -> gamma and beta channel-wise
# m_ij = gamma * F_i + beta
# F' = [F_i; mean (m_ij)]
# decoder 1280*2
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
    return {'images':nodes.mailbox['m'].mean(1)}

def edge_udf(edges):
    return {'m': edges.data['pose_gamma']*edges.src['image'] + edges.data['pose_beta']}

class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN,self).__init__()
        self.opt = opt
        self.edge_encoder = edge_encoder()

    def forward(self,g):
        with g.local_scope():
            g.edata['pose_gamma'], g.edata['pose_beta'] = self.edge_encoder(g.edata['pose'])
            g.update_all(edge_udf,node_udf)
            return g.ndata['images']
            


            



### identity 
class multi_view_dgl_identity_model(nn.Module):
    def __init__(self, opt):
        super(multi_view_dgl_identity_model, self).__init__()
        self.opt = opt
        self.encoder = encoder(self.opt)
        self.gcn = GCN_identity(self.opt)
        self.decoder = decoder(self.opt, 1280)

    def forward(self, g):
        with g.local_scope():
            image = g.ndata['image'] 
            image = image.view(-1, self.opt.camera_num, 3, self.opt.image_size, self.opt.image_size)
            h = self.encoder(image)
            g.ndata['image'] = h.view(-1, h.size()[-3], h.size()[-2], h.size()[-1])
            g = self.gcn(g)
            h = g.ndata['image']
            pred_image = self.decoder(h)
            return pred_image

def node_udf_identity(nodes):
    return {'image':nodes.data['image']}  

class GCN_identity(nn.Module):
    def __init__(self, opt):
        super(GCN,self).__init__()
        self.opt = opt
        #self.edge_encoder = edge_encoder()

    def forward(self,g):
        with g.local_scope():
            #g.edata['gamma'], g.edata['beta'] = self.edge_encoder(g.edata['pose'])
            g.update_all(fn.copy_u('image','m'),node_udf_identity)
            return g


# 