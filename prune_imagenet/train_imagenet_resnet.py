import sys
sys.path.append('./')
import os
import shutil
import random
import time
import torch
import json

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models.resnet_imagenet import *
import distributed as dist
import apex
from apex import amp
from apex.parallel import DistributedDataParallel

from utils.cal_FLOPs import print_model_param_flops


def cal_cfg_all(cfg):
    _net = resnet(arch=50, cfg=cfg).cuda(dist.get_local_rank())
    _, _, cfg_all = print_model_param_flops(_net)
    return cfg_all


def copynet_byL1(model, value:np.array):
    value = value.astype(np.int32)
    cfg = cal_cfg_all(value)
    newmodel = resnet(arch=50, cfg=value)
    count = 0
    inshape = list(range(3))
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data[inshape].clone()
            m1.bias.data = m0.bias.data[inshape].clone()
            m1.running_mean = m0.running_mean[inshape].clone()
            m1.running_var = m0.running_var[inshape].clone()
        elif isinstance(m0, nn.Conv2d):
            # cal L1
            weight_copy = m0.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:cfg[count]]
            ind = np.sort(arg_max_rev)
            # random
            # arg_max_rev = np.random.choice(list(range(m0.weight.data.shape[0])), size=cfg[count], replace=False)
            # ind = np.sort(arg_max_rev)
            if len(inshape) == 4*m0.weight.data.shape[1] or len(inshape) == 2*m0.weight.data.shape[1]:
                inshape = list(range(m0.weight.data.shape[1]))
            w1 = m0.weight.data[:, inshape, :, :].clone()
            w1 = w1[ind, :, :, :].clone()
            m1.weight.data = w1.clone() #* m0.weight.data.shape[0] / len(ind)
            inshape = ind
            count += 1
        elif isinstance(m0, nn.Linear):
            if count == len(cfg):
                m1.weight.data = m0.weight.data[:, inshape].clone()
                m1.bias.data = m0.bias.data.clone()
                count += 1
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, nn.BatchNorm1d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    # print_BN(newmodel)
    return newmodel


# train
def train(epoch, net, optimizer, train_loader, flag, log):
    net.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(dist.get_local_rank()), target.cuda(dist.get_local_rank())
        data, target = Variable(data), Variable(target)
        output = net(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            t1 = time.time()
            info = 'Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, {} mins'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, round((t1 - t0) / 60, 2))
            if flag == 1 and dist.is_primary():
                print_log(info, log)
            if flag == 2 and dist.is_primary():
                print(info)
            t0 = t1
        # torch.cuda.empty_cache()
    
    return net

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res

# test
def test(net, test_loader):
    net.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    t0 = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(dist.get_local_rank()), target.cuda(dist.get_local_rank())
            data, target = Variable(data), Variable(target)
            output = net(data)

            res = accuracy(output, target)
            correct1 += res[0]
            correct5 += res[1]
        
    t1 = time.time()
    if dist.is_primary():
        print('test time: {} mins'.format(round((t1-t0)/60, 2)))
    return correct1 / float(len(test_loader.dataset)), correct5 / float(len(test_loader.dataset))

def cal_BN(net, train_loader):
    net.eval()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            L = m.running_mean.shape
            m.running_mean = torch.zeros(L).cuda()
            m.running_var = torch.ones(L).cuda()
            m.train()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 200:
                break
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = net(data)


def test_bn(net, test_loader, train_loader):
    cal_BN(net, train_loader)
    net.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    t0 = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(dist.get_local_rank()), target.cuda(dist.get_local_rank())
            data, target = Variable(data), Variable(target)
            output = net(data)

            res = accuracy(output, target)
            correct1 += res[0]
            correct5 += res[1]
        
    t1 = time.time()
    if dist.is_primary():
        print('test time: {} mins'.format(round((t1-t0)/60, 2)))
    return correct1 / float(len(test_loader.dataset)), correct5 / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, savename):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, '{}.pth.tar'.format(savename)))

def print_log(print_string, log):
    print("{} {}".format(time.strftime("[%Y-%m-%d %H:%M:%S]"), print_string))
    log.write('{} {}\n'.format(time.strftime("[%Y-%m-%d %H:%M:%S]"), print_string))
    log.flush()


def main(args):

    torch.cuda.set_device(dist.get_local_rank())
    args.distributed = dist.get_world_size() > 1

    if dist.is_primary():
        # log
        log = open(os.path.join(args.save, '{}_{}.txt'.format(args.name, args.seed)), 'w')
        for arg in vars(args):
            print_log('{} : {}'.format(arg, getattr(args, arg)), log)
        # print_log('training renset0.4-{}'.format(args.width_mult), log)
    else: 
        log = None


    # load data
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'train'),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    train_sampler = dist.data_sampler(
        trainset,
        shuffle=True,
        distributed=args.distributed
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch_size // args.n_gpu, 
        sampler=train_sampler,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_path, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # define model
    if args.original == 'True':
        if dist.is_primary():
            print_log('train an unpruned network', log)
        net = resnet(arch=50).cuda(dist.get_local_rank())
    else:
        # value  = np.array([32, 24, 24, 64, 48, 64, 64, 224, 128, 128, 128, 128, 192, 512, 512, 512]).astype(np.int32)
        with open("prune_imagenet/pruned_arch.json", 'r') as f:
            _value = json.load(f)
        value = np.array(_value[0]['resnet50']).astype(np.int32)
        
        if args.finetune == 'True':
            assert args.pretrained, "when args.finetune is true, a pretrained model must be provided"
            if dist.is_primary():
                print_log('finetune the pruned network', log)
                print_log('pruned architecture: {}'.format(value), log)
            model_tmp = resnet(arch=50)
            ckpt = torch.load(args.pretrained, map_location=torch.device('cpu'))
            model_tmp.load_state_dict(ckpt['state_dict'])
            model_tmp = model_tmp.cuda(dist.get_local_rank())
            net = copynet_byL1(model_tmp, value).cuda(dist.get_local_rank())
        else:
            if dist.is_primary():
                print_log('retrain the pruned network', log)
                print_log('pruned architecture: {}'.format(value), log)
            net = resnet(arch=50, cfg=value).cuda(dist.get_local_rank())

    prec = test(net, test_loader)
    prec_bn = test_bn(net, test_loader, train_loader)

    if dist.is_primary():
        print_log('Ori acc of optimal pruned result: {}'.format(prec), log)
        print_log('Acc of optimal pruned result after BN calibration: {}'.format(prec_bn), log)

    if args.sync_bn:
        net = apex.parallel.convert_syncbn_model(net)
    
    # optimizer
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)

    # amp
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.amp)
    net = DistributedDataParallel(net)

    lr = args.lr
    best_prec1 = 0.
    best_prec5 = 0.
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if epoch in [int(args.epochs*1/3), int(args.epochs*2/3)]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                lr *= 0.1
        if dist.is_primary():    
            print_log('epoch: {}, lr: {}'.format(epoch, lr), log)
        train(epoch, net, optimizer, train_loader, flag=1, log=log)
        prec1, prec5 = test(net, test_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = prec5 if is_best else best_prec5
        if dist.is_primary():
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict(),
                'best_prec1': best_prec1,
                'prec5': best_prec5,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=args.save, savename='{}_best'.format(args.name))

            print_log('best_prec1: {}, best_prec5: {}'.format(best_prec1, best_prec5), log)


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument('--dist_url', default=f"tcp://127.0.0.1:{port}")
    parser.add_argument('--amp', type=str, default='O1')

    parser.add_argument('--data_path', type=str, default='/data/llb/imagenet', help='input data path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--save', type=str, default='./result/mobilenet')
    parser.add_argument('--name', type=str, default='imagenet_resnet')
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--finetune', type=str, choices=['True', 'False'])
    parser.add_argument('--original', type=str, choices=['True', 'False'])

    args = parser.parse_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
    
