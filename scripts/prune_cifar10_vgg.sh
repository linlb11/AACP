python prune_cifar/prune_cifar10_vgg.py \
--pretrained=pretrained/vgg_best.pth.tar \
--save=./result/cifar/vgg \
--device=0 \
--iters=200 \
--log_name='vgg_param_0.5_flops_0.5_L1' \
--param_prune_rate=0.5 \
--FLOPs_prune_rate=0.5 \
--seed=0