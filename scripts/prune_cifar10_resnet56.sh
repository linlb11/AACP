python prune_cifar/prune_cifar10_resnet.py \
--arch=56 \
--pretrained=pretrained/resnet56_best.pth.tar \
--save=./result/cifar/resnet56 \
--device=0 \
--iters=200 \
--log_name='resnet_param_1_flops_0.5_L1' \
--param_prune_rate=1 \
--FLOPs_prune_rate=0.5 \
--seed=0