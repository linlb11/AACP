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
from utils.cal_FLOPs import print_model_param_flops
from models.mobilenet_imagenet import *

# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/mnt/llb/imagenet', help='input data path')
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--save', type=str, default='./result/mobilenet')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--pretrained', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_name', type=str)
parser.add_argument('--param_prune_rate', type=float)
parser.add_argument('--FLOPs_prune_rate', type=float)
parser.add_argument('--iters', type=int)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# log
log = open(os.path.join(args.save, '{}_{}.txt'.format(args.log_name, args.seed)), 'w')

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

# load data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'train'),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ), 
    batch_size=args.batch_size, shuffle=True, num_workers=8)

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
    batch_size=args.batch_size, shuffle=False, num_workers=8)

# training model
def train(epoch, net, optimizer, flag):
    net.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            t1 = time.time()
            info = 'Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, {} mins'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, round((t1 - t0) / 60, 2))
            if flag == 1:
                print_log(info, log)
            if flag == 2:
                print(info)
            t0 = t1

    return net

# testing model
def test_ori(net):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
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
            if batch_idx > 150:
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
    def __init__(self, Dim:int, NP:int, alpha:float, cfg:list, T:int, prune_step:list, param_prune_rate:float=0.0, FLOPs_prune_rate:float=0.0, F:float=0.5, CR:float=0.8):
        self.F = F # scaling factor
        self.CR = CR # crossover probability
        self.Dim = Dim # dimention
        self.NP = NP # population number
        self.T = T # iteration times
        self.fitness_list = []
        
        assert 0 < alpha <= 1, 'alpha should be in (0, 1]'
        self.alpha = alpha # each conv layer can keep up to alpha*100% channels
        self.cfg = cfg # configuration of original model
        self.cfg_all = self.cal_cfg_all(cfg)

        self.max_stay = 4 # 
        self.prune_step = prune_step # interval of channel search space
        # self.output_shape = output_shape # shape of feature map
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
        self.remain_flops = self.cal_flops(self.cfg) * self.FLOPs_prune_rate

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

        print('flops: ', self.cal_flops(self.cfg), 'G')

    def cal_cfg_all(self, cfg):
        # _net = resnet(arch=args.arch, cfg=cfg).cuda()
        _net = mobilenet_v2(cfg=cfg, width_mult=args.width_mult).cuda()
        _, _, cfg_all = print_model_param_flops(_net)
        return cfg_all
    
    def cal_flops(self, cfg):
        # _net = resnet(arch=args.arch, cfg=cfg).cuda()
        _net = mobilenet_v2(cfg=cfg, width_mult=args.width_mult).cuda()
        flops, _, _ = print_model_param_flops(_net)
        return flops

    def copynet_byL1(self, model, value:np.array):
        value = np.array(value).astype(np.int)
        assert len(value) == len(self.cfg), 'len(value) != len(self.cfg)'
        cfg = self.cal_cfg_all(value)
        # newmodel = resnet(arch=args.arch, cfg=value)
        newmodel = mobilenet_v2(cfg=value, width_mult=args.width_mult)
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

                if m0.weight.data.size(1) == 1 and m0.weight.data.size(2) == 3:
                    m1.weight.data = m0.weight.data[inshape].clone()
                    count += 1
                    continue

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
        # net_single = resnet(arch=args.arch)
        net_single = mobilenet_v2(width_mult=args.width_mult)
        net_single = net_single.cuda()
        net_single.load_state_dict(net.module.state_dict())

        net_pruned = self.copynet_byL1(model=net_single, value=value)

        # test network
        net_pruned = net_pruned.cuda()
        if torch.cuda.device_count() > 1:
            net_pruned = torch.nn.DataParallel(net_pruned)
        acc = test(net_pruned)
        return acc

    def init(self):
        gen = []
        for j in range(self.Dim):
            index = np.random.choice(self.search_space[j], size=1)
            gen.append(index[0])
        
        gen_all = self.cal_cfg_all(gen)
        current_channels = np.sum(gen_all)
        current_flops = self.cal_flops(gen)
        while(current_channels > self.remain_channels or current_flops > self.remain_flops):
            # # reinit
            # gen = []
            # for j in range(self.Dim):
            #     index = np.random.choice(self.search_space[j], size=1)
            #     gen.append(index[0])
            # gen_all = self.cal_cfg_all(gen)
            # current_channels = np.sum(gen_all)
            # current_flops = self.cal_flops(gen)
            
            # random
            r = np.random.choice(list(range(len(gen))), size=1)
            if gen[r[0]] > self.prune_step[r[0]]:
                gen[r[0]] -= self.prune_step[r[0]]
                gen_all = self.cal_cfg_all(gen)
                current_channels = np.sum(gen_all)
                current_flops = self.cal_flops(gen)
        
        return(np.array(gen))

    def Boundary(self, mut:np.array):
        mut_new = copy.deepcopy(mut.astype(np.int))
        assert mut_new.shape[0] == self.Dim
        for i in range(self.Dim):
            mut_new[i] = round(mut_new[i] / self.prune_step[i]) * self.prune_step[i]
            if mut_new[i] > self.upperbound[i]:
                mut_new[i] = self.upperbound[i]
            elif mut_new[i] < self.prune_step[i]:
                mut_new[i] = self.prune_step[i]

        mut_new_all = self.cal_cfg_all(mut_new)
        current_channels = np.sum(mut_new_all)
        current_flops = self.cal_flops(mut_new)
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
                current_flops = self.cal_flops(mut_new)

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
            mutation_ = (self.best_unit.gene + self.F * (self.group[r2].gene - self.group[r3].gene)).astype(np.int)
            mutation_ = (self.Boundary(mutation_)).astype(np.int)
            
            # crossover
            # self.calprobability()
            randj = random.randint(0, self.Dim - 1)
            for j in range(self.Dim):
                # if random.random() <= self.group[i].rfitness or j == randj:
                if random.random() <= self.CR or j == randj:
                    crossover_[j] = mutation_[j]
                else:
                    crossover_[j] = self.group[i].gene[j]
            crossover_ = (self.Boundary(crossover_)).astype(np.int)

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
            print_log('t={}, best fitness: {}, time: {:.2f}s'.format(t, self.best_unit.fitness, end_time-start_time), log)
            p = list(self.best_unit.gene / np.array(self.cfg))
            p = [round(i, 2) for i in p]
            print_log('flops: {} / {} = {}, channels: {} / {} = {}'.format(
                self.cal_flops(self.best_unit.gene), 
                self.cal_flops(self.cfg),
                self.cal_flops(self.best_unit.gene) / self.cal_flops(self.cfg),
                np.sum(self.cal_cfg_all(self.best_unit.gene)), 
                np.sum(self.cfg_all),
                np.sum(self.cal_cfg_all(self.best_unit.gene)) / np.sum(self.cfg_all)
            ), log)
            print_log('best choice: {}'.format(self.best_unit.gene), log)
            print_log('\t{}\n'.format(p), log)
            t += 1

        cfg_pruned = self.best_unit.gene.astype(np.int)

        # net_s = resnet(arch=args.arch)
        net_s = mobilenet_v2(width_mult=args.width_mult)
        net_s = net_s.cuda()
        net_s.load_state_dict(net.module.state_dict())
        net_pruned = self.copynet_byL1(model=net_s, value=cfg_pruned)

        return net_pruned, cfg_pruned
            
            
    def plot(self):
        flist = []
        for i in range(len(self.fitness_list)):
            flist.append(round(self.fitness_list[i].numpy(), 4))
        print_log(flist, log)


if __name__ == "__main__":

    for arg in vars(args):
        print_log('{} : {}'.format(arg, getattr(args, arg)), log)
    
    # load model
    net = mobilenet_v2(width_mult=args.width_mult).cuda()
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    prec = test_ori(net)
    print_log('original model accuracy: {}'.format(prec), log)
    # cfg = list(np.array([16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160]) * 6)
    cfg = [32, 16*6, 24*6, 24*6, 32*6, 32*6, 32*6, 64*6, 64*6, 64*6, 64*6, 96*6, 96*6, 96*6, 160*6, 160*6, 160*6, 16, 24, 32, 64, 96, 160, 320]
    step = [4, 2*6,  3*6,  3*6,  4*6,  4*6,  4*6,  8*6,  8*6,  8*6,  8*6,  12*6, 12*6, 12*6,  20*6,  20*6, 20*6,  2,  3,  4,  8,  12, 20,  40]

    de1 = DE(
        Dim=len(cfg), 
        NP=10, 
        alpha=args.alpha, 
        cfg=cfg, 
        T=args.iters, 
        prune_step=step,
        param_prune_rate=args.param_prune_rate, 
        FLOPs_prune_rate=args.FLOPs_prune_rate, 
        F=0.5, 
        CR=0.8
    )
    net_pruned, cfg_pruned = de1.run_DE()
    net_pruned = net_pruned.cuda()
    if torch.cuda.device_count() > 1:
        net_pruned = torch.nn.DataParallel(net_pruned)

    print_log('\n\nOptimal pruned result: {}\n\n'.format(cfg_pruned), log)
    print_log('Ori acc of optimal pruned result: {}'.format(test_ori(net_pruned)), log)
    print_log('Acc of optimal pruned result after BN calibration: {}'.format(test(net_pruned)), log)
    print_log('Finished !! Please update the optimal pruned result(mobilenetv2) in pruned_arch.json', log)
    

