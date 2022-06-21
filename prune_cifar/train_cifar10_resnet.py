import sys
sys.path.append('./')
import os
import numpy as np
import random
import argparse
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from models.resnet_cifar import *


# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data', help='input data path')
parser.add_argument('--arch', type=int, default=56, help='input network architecture')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--save', type=str, default='./result')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_name', type=str)
parser.add_argument('--device', type=int)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# log
log = open(os.path.join(args.save, 'cifar10_160_{}_{}.txt'.format(args.log_name, args.seed)), 'w')

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

# set GPU 
torch.cuda.set_device(args.device)

# load data
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
else:
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

# training model
def train(epoch, net, optimizer):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            info = 'Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data)
            print_log(info, log)
    return net

# testing model
def test_ori(net):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    test_loss /= len(test_loader.dataset)
    info = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(info)
    # print_log(info, log)
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, save_name):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, '{}.pth.tar'.format(save_name)))

def print_log(print_string, log):
    print("{} {}".format(time.strftime("[%Y-%m-%d %H:%M:%S]"), print_string))
    log.write('{} {}\n'.format(time.strftime("[%Y-%m-%d %H:%M:%S]"), print_string))
    log.flush()


if __name__ == "__main__":

    for arg in vars(args):
        print_log('{} : {}'.format(arg, getattr(args, arg)), log)

    # define network
    net = resnet(dataset='cifar10', depth=args.arch)
    net = net.cuda()

    # optimazer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # train
    lr = args.lr
    best_prec1 = 0.
    for epoch in range(args.epochs):
        if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                lr *= 0.1
        train(epoch, net, optimizer)
        prec1 = test_ori(net)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg':net.cfg
        }, is_best, filepath=args.save, save_name='resnet56_best')
        print_log('lr: {}, best_prec1: {}'.format(lr, best_prec1), log)