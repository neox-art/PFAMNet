import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
import os
import random
import torch
from dataset import DatasetField
from losses import *
from models import PFAMNet
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import visualize, create_train_arg_parser, evaluate
from sklearn.model_selection import train_test_split

def flatten_grads(grads, params):
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).view(-1))
        else:
            flat.append(g.view(-1))
    return torch.cat(flat)

def to_scalar(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.item()
    return float(x)

def define_loss(model_type):
    if model_type == 'field':
        criterion = LossF()
    return criterion

def build_model(model_type, use_deep_supervision=True):
    if model_type == 'field':
        model = PFAMNet(use_deep_supervision=use_deep_supervision)
    return model

def train_model(model, inputs, targets, criterion, optimizer, model_type, lossweighter=None):
    optimizer.zero_grad()
    outputs = model(inputs)
    task_losses = criterion(outputs, targets)
    if lossweighter is not None:
        total_loss, raw_log_dict = lossweighter(task_losses)
        log_dict = {'total': total_loss.detach()}
        for k, v in raw_log_dict.items():
            if k == 'total':
                continue
            if torch.is_tensor(v):
                log_dict[k] = v.detach()
            else:
                log_dict[k] = v
    elif model_type in ['field']:
        loss_sem = task_losses['sem']
        loss_edge = task_losses['edge']
        loss_dist = task_losses['dist']
        total_loss = loss_sem + loss_edge + loss_dist
        log_dict = {'total': total_loss.detach(), 'sem': loss_sem.detach(), 'edge': loss_edge.detach(), 'dist': loss_dist.detach()}
    else:
        raise ValueError(f'Unsupported model_type: {model_type}')
    total_loss.backward()
    optimizer.step()
    return log_dict
if __name__ == '__main__':
    args = create_train_arg_parser().parse_args()
    args.lossweighter_type = 'uncertaintyweight'
    args.train_path = '/workspace/Dataset/ai4full/train/image'
    args.val_path = '/workspace/Dataset/ai4full/val/image'
    args.model_type = 'field'
    args.save_path = '/workspace/Experiment/pfamnet_pfe_amff_ai4full'
    CUDA_SELECT = 'cuda:{}'.format(args.cuda_no)
    log_path = args.save_path + '/summary'
    writer = SummaryWriter(log_dir=log_path)
    train_file_names = glob.glob(os.path.join(args.train_path, '*.tif'))
    random.seed(41)
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, '*.tif'))
    img_train = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    img_val = [os.path.splitext(os.path.basename(p))[0] for p in val_file_names]
    train_file, val_file = (img_train, img_val)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else 'cpu')
    print(device)
    model = build_model(args.model_type, args.use_deep_supervision)
    if torch.cuda.device_count() > 2:
        print("Let's use", torch.cuda.device_count(), 'GPUs!')
        model = nn.DataParallel(model)
    model = model.to(device)
    epoch_start = '0'
    if args.model_type in ['field']:
        task_names = ['sem', 'edge', 'dist']
    else:
        raise ValueError(f'Unsupported model_type: {args.model_type}')
    if args.lossweighter_type == 'uncertaintyweight':
        lossweighter = UncertaintyWeightedMultiTaskLoss(task_names).to(device)
    else:
        lossweighter = None
    if args.use_pretrained:
        print('Loading Model {}'.format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split('.')[0]
        print(epoch_start)
    print('train', args.use_pretrained)
    trainLoader = DataLoader(DatasetField(args.train_path, train_file, model_type=args.model_type), batch_size=args.batch_size, drop_last=False, shuffle=True)
    devLoader = DataLoader(DatasetField(args.val_path, val_file, model_type=args.model_type), drop_last=False)
    displayLoader = DataLoader(DatasetField(args.val_path, val_file, model_type=args.model_type), batch_size=args.val_batch_size, drop_last=False, shuffle=True)
    base_lr = 5e-05
    min_lr = 1e-06
    warmup_epochs = 3
    if lossweighter is not None:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(lossweighter.parameters()), lr=base_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs), torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=min_lr)], milestones=[warmup_epochs])
    criterion = define_loss(args.model_type)
    print('Device:', device)
    print('Training epochs:', args.num_epochs)
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_path, 'best_model.pt')
    for epoch in tqdm(range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)):
        global_step = (epoch - 1) * len(trainLoader)
        running_loss = 0.0
        for i, batch in enumerate(tqdm(trainLoader)):
            step = global_step + i
            model.train()
            img_file_name = batch[0]
            inputs = batch[1].to(device)
            targets = [x.to(device) for x in batch[2:]]
            loss_dict = train_model(model=model, inputs=inputs, targets=targets, criterion=criterion, optimizer=optimizer, model_type=args.model_type, lossweighter=lossweighter)
            writer.add_scalar('loss/total', to_scalar(loss_dict['total']), step)
            if loss_dict.get('sem') is not None:
                writer.add_scalar('loss/sem', to_scalar(loss_dict['sem']), step)
            if loss_dict.get('edge') is not None:
                writer.add_scalar('loss/edge', to_scalar(loss_dict['edge']), step)
            if loss_dict.get('dist') is not None:
                writer.add_scalar('loss/dist', to_scalar(loss_dict['dist']), step)
            if lossweighter is not None:
                if loss_dict.get('sem_weight') is not None:
                    writer.add_scalar('weight/sem', to_scalar(loss_dict['sem_weight']), step)
                if loss_dict.get('edge_weight') is not None:
                    writer.add_scalar('weight/edge', to_scalar(loss_dict['edge_weight']), step)
                if loss_dict.get('dist_weight') is not None:
                    writer.add_scalar('weight/dist', to_scalar(loss_dict['dist_weight']), step)
            running_loss += to_scalar(loss_dict['total']) * inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_file_names)
        print(epoch_loss)
        if epoch % 1 == 0:
            dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer, model_type=args.model_type)
            writer.add_scalar('loss_valid', dev_loss, epoch)
            print('Global Loss:{} Val Loss:{}'.format(epoch_loss, dev_loss))
            if dev_loss < best_val_loss:
                best_val_loss = dev_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved at epoch {epoch} with val_loss {dev_loss:.6f}')
        else:
            print('Global Loss:{} '.format(epoch_loss))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, str(epoch) + '.pt'))
