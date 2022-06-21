python -W ignore prune_imagenet/prune_imagenet_resnet.py \
--data_path=/root/autodl-tmp/data/imagenet \
--pretrained=pretrained/imagenet_resnet50_best.pth.tar \
--save=./result/imagenet/resnet50 \
--iters=200 \
--arch=50 \
--log_name='imagenet_resnet_param_1_flops_0.49_iter200' \
--param_prune_rate=1 \
--FLOPs_prune_rate=0.49 \
--seed=0
