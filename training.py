import torch, numpy, argparse, os, time, math, random
from dataloader import MRPGDataSet
import torch.nn.functional as F
import torch.optim as optim
from model import models
import utils

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='airsim-mrmps-data')
parser.add_argument('-target', type=str, default='train')
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-dropout', type=float, default=0.2, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.01)
#parser.add_argument('-encoder_name', type=str, default="resnet34")
parser.add_argument('-npose', type=int, default=8)
parser.add_argument('-model_dir', type=str, default="trained_models")
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-device', type=str, default="cuda:0")
parser.add_argument('-model', type=str, default="single_view")
parser.add_argument('-camera_num', type=int, default=5)
parser.add_argument('-pretrained', action="store_true")
opt = parser.parse_args()

def compute_MSE_loss(targets, predictions, reduction='mean'):
    target_depths = targets
    pred_depths = predictions.view(-1, opt.camera_num, 3, opt.image_size, opt.image_size)
    loss = F.mse_loss(pred_depths, target_depths, reduction=reduction)
    return loss

def train(model, device, dataloader, optimizer, epoch, log_interval=50):
    model.train()
    train_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        images, poses, depths = data
        images, poses, depths = images.to(device), poses.to(device), depths.to(device)
        pred_depth = model(images,poses)
        loss = compute_MSE_loss(depths, pred_depth)
        train_loss += loss
        # VAEs get NaN loss sometimes, so check for it
        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            optimizer.step()
        batch_num+=1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * opt.batch_size, len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
    avg_train_loss = train_loss/batch_num
    return [avg_train_loss]
def test(model, device, dataloader):
    model.eval()
    test_loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            images, poses, depths = data
            images, poses, depths = images.to(device), poses.to(device), depths.to(device)
            pred_depth = model(images, poses)
            test_loss += compute_MSE_loss(depths, pred_depth)
            batch_num+=1
    avg_test_loss = test_loss/batch_num
    return [avg_test_loss]

if __name__ == '__main__':
    os.system('mkdir -p ' + opt.model_dir)

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # define colored_lane symbol for dataloader
    dataset = MRPGDataSet(opt)
    trainset, valset = torch.utils.data.random_split(dataset,
                                                [int(0.90 * len(dataset)), len(dataset) - int(0.90 * len(dataset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    # define model file name
    opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-lrt={opt.lrt}'
    opt.model_file += f'-seed={opt.seed}'
    print(f'[will save model as: {opt.model_file}]')
    mfile = opt.model_file + '.model'

    # load previous checkpoint or create new model
    if os.path.isfile(mfile):
        print(f'[loading previous checkpoint: {mfile}]')
        checkpoint = torch.load(mfile)
        model = checkpoint['model']
        model.cuda()
        optimizer = optim.Adam(model.parameters(), opt.lrt)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        scheduler.load_state_dict(checkpoint['scheduler'])
        n_iter = checkpoint['n_iter']
        utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
    else:
        model = models.single_view_model(opt)
        optimizer = optim.Adam(model.parameters(), opt.lrt)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        n_iter = 0

    stats = torch.load(opt.dataset + '/data_stats.pth')
    model.cuda()
    print('[training]')
    for epoch in range(1000):
        t0 = time.time()
        train_losses = train(model,opt.device, trainloader,optimizer,epoch)
        val_losses = test(model,opt.device, trainloader)
        scheduler.step()
        n_iter += 1
        model.cpu()
        torch.save({'model': model,
                    'optimizer': optimizer.state_dict(),
                    'n_iter': n_iter,
                    'scheduler': scheduler.state_dict()}, opt.model_file + '.model')
        model.cuda()
        log_string = f'step {n_iter} | '
        log_string += utils.format_losses(*train_losses, split='train')
        log_string += utils.format_losses(*val_losses, split='valid')
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)
