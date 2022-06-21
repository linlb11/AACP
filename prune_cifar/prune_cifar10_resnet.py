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
import copy
import time
from torch.autograd import Variable
from models.resnet_cifar import *


# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data', help='input data path')
parser.add_argument('--arch', type=int, default=56)
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
parser.add_argument('--param_prune_rate', type=float)
parser.add_argument('--FLOPs_prune_rate', type=float)
parser.add_argument('--device', type=int)
parser.add_argument('--istrain', type=int)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--iters', type=int)
parser.add_argument('--finetune_or_retrain', type=str, default='finetune', choices=['finetune', 'retrain'])
parser.add_argument('--cal_BN_iters', type=int, default=100)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# log
log = open(os.path.join(args.save, 'cifar_resnet_{}_{}_scratch.txt'.format(args.log_name, args.seed)), 'w')

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
def train(epoch, net, optimizer, flag):
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
            if flag == 1:
                print_log(info, log)
            if flag == 2:
                print(info)
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

# testing model
def cal_BN(net):
    net.eval()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            L = m.running_mean.shape
            m.running_mean = torch.zeros(L).cuda()
            m.running_var = torch.ones(L).cuda()
            m.train()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > args.cal_BN_iters:
                break
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = net(data)
        
def test(net):
    cal_BN(net)
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
    
class unit:
    def __init__(self, gene:np.array):
        self.gene = gene
        self.stay = 0
        self.fitness = 0.
        self.rfitness = 0.

class DE:
    def __init__(self, Dim:int, NP:int, alpha:float, cfg:list, T:int, prune_step:list, output_shape:list, finetune_or_retrain:str, param_prune_rate:float=0.0, FLOPs_prune_rate:float=0.0, F:float=0.5, CR:float=0.8):
        self.F = F # scaling factor
        self.CR = CR # crossover probability
        self.Dim = Dim # dimention
        self.NP = NP # population number
        self.T = T # iteration times
        self.finetune_or_retrain = finetune_or_retrain
        self.fitness_list = []
        
        assert 0 < alpha <= 1, 'alpha should be in (0, 1]'
        self.alpha = alpha # each conv layer can keep up to alpha*100% channels
        self.cfg = cfg # configuration of original model
        self.cfg_all = self.cal_cfg_all(cfg)

        self.max_stay = 4 # 
        self.prune_step = prune_step # interval of channel search space
        self.output_shape = output_shape # shape of feature map
        self.param_prune_rate = param_prune_rate # pruning rate in number of channels
        self.FLOPs_prune_rate = FLOPs_prune_rate # pruning rate in number of FLOPs
        self.upperbound = []
        self.totalchannels = np.sum(self.cfg_all)
        
        self.search_space = []

        for i in range(len(self.cfg)):
            upper = int(int(self.cfg[i] * self.alpha) / self.prune_step[i]) * self.prune_step[i]
            self.upperbound.append(upper)
            self.search_space.append(list(range(self.prune_step[i], upper+1, self.prune_step[i])))

        self.remain_channels = int(self.totalchannels * self.param_prune_rate)
        self.remain_flops = self.cal_flops(self.cfg_all) * self.FLOPs_prune_rate

        print('cfg_all', self.cfg_all)
        print('sum of cfg_all', np.sum(self.cfg_all))
        print('remain channels', self.remain_channels)
        # initialize the group gene
        print_log('begin initialize: ', log)
        self.group = []
        for i in range(NP):
            print('init, i: {}'.format(i))
            self.group.append(unit(self.init()))
       
        # initialize the group fitness
        for i in range(NP):
            t0 = time.time()
            self.group[i].fitness = self.Fitness(self.group[i].gene)
            t1 = time.time()

        print_log('finish initialize!', log)
        self.best_unit = copy.deepcopy(self.group[0])

        assert len(self.cfg) == len(self.prune_step), 'len(self.cfg) != len(self.step)'
        assert len(self.output_shape) == len(self.cfg_all), 'len(self.output_shape) != len(self.cfg_all)'

        print('flops: ', self.cal_flops(self.cfg_all), 'G')
        net = resnet(dataset='cifar10', depth=args.arch, cfg=cfg)
        net = net.cuda()

    def cal_cfg_all(self, cfg):
        cfg_all = [16]
        assert len(cfg) == self.Dim, 'len(cfg) != self.Dim'
        for i in range(self.Dim * 2):
            if i // (self.Dim * 2 / 3) == 0:
                c = 16
            elif i // (self.Dim * 2 / 3) == 1:
                c = 32
            else:
                c = 64
            if i % 2 == 1:
                cfg_all.append(c)
            else:
                cfg_all.append(cfg[int(i/2)])
        return cfg_all

    def cal_flops(self, cfg):
        assert len(cfg) == len(self.cfg_all)
        flops = []
        for i in range(len(cfg)):
            if i == 0:
                in_channel = 3
            else:
                in_channel = cfg[i-1]
            
            flops.append(in_channel * 3 * 3 * cfg[i] * self.output_shape[i] * self.output_shape[i])
        self.flops = flops
        return sum(flops) / 1e9

    def copynet_byL1(self, model, value:np.array):
        value = value.astype(np.int32)
        assert len(value) == len(self.cfg), 'len(value) != len(self.cfg)'
        cfg = self.cal_cfg_all(value)
        newmodel = resnet(dataset='cifar10', depth=args.arch, cfg=value)
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
                # print(m0.weight.data.shape, len(ind), len(inshape))
                # copy
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

        return newmodel

    def Fitness(self, value:np.array):

        # initialize the weights of net_pruned by the weights of net
        net_pruned = self.copynet_byL1(model=net, value=value)
        acc = test(net_pruned)
        return acc

    def init(self):
        gen = []
        for j in range(self.Dim):
            index = np.random.choice(self.search_space[j], size=1)
            gen.append(index[0])
        
        gen_all = self.cal_cfg_all(gen)
        current_channels = np.sum(gen_all)
        current_flops = self.cal_flops(gen_all)
        while(current_channels > self.remain_channels or current_flops > self.remain_flops):
            # # reinit
            # gen = []
            # for j in range(self.Dim):
            #     index = np.random.choice(self.search_space[j], size=1)
            #     gen.append(index[0])
            # gen_all = self.cal_cfg_all(gen)
            # current_channels = np.sum(gen_all)
            # current_flops = self.cal_flops(gen_all)
            # random
            r = np.random.choice(list(range(len(gen))), size=1)
            if gen[r[0]] > self.prune_step[r[0]]:
                gen[r[0]] -= self.prune_step[r[0]]

                gen_all = self.cal_cfg_all(gen)
                current_channels = np.sum(gen_all)
                current_flops = self.cal_flops(gen_all)
        
        return(np.array(gen))

    def Boundary(self, mut:np.array):
        mut_new = copy.deepcopy(mut)
        assert mut_new.shape[0] == self.Dim
        for i in range(self.Dim):
            mut_new[i] = round(mut_new[i] / self.prune_step[i]) * self.prune_step[i]
            if mut_new[i] > self.upperbound[i]:
                mut_new[i] = self.upperbound[i]
            elif mut_new[i] < self.prune_step[i]:
                mut_new[i] = self.prune_step[i]

        mut_new_all = self.cal_cfg_all(mut_new)
        current_channels = np.sum(mut_new_all)
        current_flops = self.cal_flops(mut_new_all)
        while(current_channels > self.remain_channels or current_flops > self.remain_flops):
            # # max
            # r = np.argsort(mut_new)[::-1][0]
            # mut_new[r] -= self.prune_step[r]

            # mut_new_all = self.cal_cfg_all(mut_new)
            # current_channels = np.sum(mut_new_all)
            # current_flops = self.cal_flops(mut_new_all)
            
            # random
            r = np.random.choice(list(range(mut_new.size)), size=1)
            if mut_new[r[0]] > self.prune_step[r[0]]:
                mut_new[r[0]] -= self.prune_step[r[0]]

                mut_new_all = self.cal_cfg_all(mut_new)
                current_channels = np.sum(mut_new_all)
                current_flops = self.cal_flops(mut_new_all)

        return mut_new

    def calprobability(self):
        maxfit = self.group[0].fitness
        for i in range(1, self.NP):
            if self.group[i].fitness > maxfit:
                maxfit = self.group[i].fitness
        for i in range(self.NP):
            self.group[i].rfitness = (0.4 * (self.group[i].fitness / maxfit) + 0.5)


    def run_one(self):
        # choose three random unit
        mutation_ = np.zeros((self.Dim,))
        crossover_ = np.zeros((self.Dim,))
        for i in range(self.NP):
            # t0 = time.time()
            # reinitialize
            if self.group[i].stay >= self.max_stay:
                self.group[i].gene = copy.deepcopy(self.init())
                self.group[i].fitness = self.Fitness(self.group[i].gene)
                self.group[i].stay = 0

            # mutation
            r1, r2, r3 = np.random.choice(list(range(i))+list(range(i+1, self.NP)), size=3, replace=False)
            mutation_ = (self.best_unit.gene + self.F * (self.group[r2].gene - self.group[r3].gene)).astype(np.int32)
            mutation_ = (self.Boundary(mutation_)).astype(np.int32)
            
            # crossover
            # self.calprobability()
            randj = random.randint(0, self.Dim - 1)
            for j in range(self.Dim):
                # if random.random() <= self.group[i].rfitness or j == randj:
                if random.random() <= self.CR or j == randj:
                    crossover_[j] = mutation_[j]
                else:
                    crossover_[j] = self.group[i].gene[j]
            crossover_ = (self.Boundary(crossover_)).astype(np.int32)

            # select
            cross_fitness = self.Fitness(crossover_)
            if cross_fitness > self.group[i].fitness:
                self.group[i].gene = copy.deepcopy(crossover_)
                self.group[i].fitness = cross_fitness
                self.group[i].stay = 0
            else:
                self.group[i].stay += 1

            # check if self.group[i, :] is the best choice
            if cross_fitness > self.best_unit.fitness:
                self.best_unit.gene = copy.deepcopy(self.group[i].gene)
                self.best_unit.fitness = self.group[i].fitness
                self.best_unit.stay = self.group[i].stay
            # t1 = time.time()
            # print_log('{} / {}, time: {:.2f}s'.format(i, self.NP, t1-t0), log)

    
    def run_DE(self):
        t = 0
        while(t < self.T):
            start_time = time.time()
            self.run_one()
            end_time = time.time()

            # record the best fitness of this round 
            self.fitness_list.append(self.best_unit.fitness)
            print_log('\nt={}, best fitness: {}, time: {:.2f}s'.format(t, self.best_unit.fitness, end_time-start_time), log)
            p = list(self.best_unit.gene / np.array(self.cfg))
            p = [round(i, 2) for i in p]
            print_log('flops: {} / {} = {}, channels: {} / {} = {}'.format(
                self.cal_flops(self.cal_cfg_all(self.best_unit.gene)), 
                self.cal_flops(self.cfg_all),
                self.cal_flops(self.cal_cfg_all(self.best_unit.gene)) / self.cal_flops(self.cfg_all),
                np.sum(self.cal_cfg_all(self.best_unit.gene)), 
                np.sum(self.cfg_all),
                np.sum(self.cal_cfg_all(self.best_unit.gene)) / np.sum(self.cfg_all)
            ), log)
            print_log('best choice: {}'.format(self.best_unit.gene), log)
            print_log('\t{}\n'.format(p), log)
            t += 1

        cfg_pruned = self.best_unit.gene.astype(np.int32)
        if self.finetune_or_retrain == 'finetune':
            # finetune
            net_pruned = self.copynet_byL1(model=net, value=cfg_pruned)   
        else:
            # retrain
            net_pruned = resnet(dataset='cifar10', depth=args.arch, cfg=cfg_pruned)

        return net_pruned, cfg_pruned


if __name__ == "__main__":

    for arg in vars(args):
        print_log('{} : {}'.format(arg, getattr(args, arg)), log)

    # define network
    net = resnet(dataset='cifar10', depth=args.arch)
    net = net.cuda()
    print_log('resnet-{} has been constructed'.format(args.arch), log)
    #optimazer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if not args.pretrained:
        best_prec1 = 0.
        for epoch in range(args.epochs):
            if epoch in [args.epochs*0.5, args.epochs*0.75]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train(epoch, net, optimizer, flag=2)
            prec1 = test_ori(net)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'cfg':net.cfg
            }, is_best, filepath=args.save, save_name='resnet{}_best'.format(args.arch))
            print('best_prec1: {}'.format(best_prec1))
        print_log('best_prec1: {}'.format(best_prec1), log)

    # load model
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['state_dict'])

    prec = test_ori(net)
    print_log('original model accuracy: {}'.format(prec), log)

    n = (args.arch - 2) // 6
    cfg = [[16]*n, [32]*n, [64]*n]
    cfg = [item for sub_list in cfg for item in sub_list]

    step = [[2]*n, [4]*n, [8]*n]
    step = [item for sub_list in step for item in sub_list]
    
    output_shape = [[32]*(2*n+1), [16]*2*n, [8]*2*n]
    output_shape = [item for sub_list in output_shape for item in sub_list]

    de1 = DE(
        Dim=3*n, 
        NP=10, 
        alpha=1, 
        cfg=cfg, 
        T=args.iters, 
        prune_step=step,
        output_shape = output_shape,
        finetune_or_retrain=args.finetune_or_retrain,
        param_prune_rate=args.param_prune_rate, 
        FLOPs_prune_rate=args.FLOPs_prune_rate, 
        F=0.5, 
        CR=0.8
    )
    net_pruned, cfg_pruned= de1.run_DE()
    net_pruned = net_pruned.cuda()

    print_log('\n\nOptimal pruned result: {}\n\n'.format(cfg_pruned), log)
    print_log('Ori acc of optimal pruned result: {}\n\n'.format(test_ori(net_pruned)), log)
    print_log('Acc of optimal pruned result after BN calibration: {}\n\n'.format(test(net_pruned)), log)
    
    # training pruned network

    # finetune
    if args.finetune_or_retrain == 'finetune':
        args.lr *= 0.1

    # retrain
    else:
        # args.epochs = int(160 / args.FLOPs_prune_rate)
        args.epochs = 320

    # optimizer
    optimizer1 = optim.SGD(net_pruned.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_prec1 = 0.0

    print_log('args.epochs:{}'.format(args.epochs), log)
    lr = args.lr
    for epoch in range(args.epochs):
        if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
            for param_group in optimizer1.param_groups:
                param_group['lr'] *= 0.1
                lr *= 0.1
        print_log('epoch: {}， lr: {}'.format(epoch, lr), log)
        train(epoch, net_pruned, optimizer1, flag=1)
        prec1 = test_ori(net_pruned)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        ### uncomment the following lines if you want to save the checkpoints
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': net_pruned.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        #     'cfg':net_pruned.cfg
        # }, is_best, filepath=args.save, save_name='prune_best')
        print_log('best_prec1: {}'.format(best_prec1), log)

