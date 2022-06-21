python -W ignore prune_imagenet/prune_imagenet_mobilenet.py \
--data_path=/root/autodl-tmp/data/imagenet \
--pretrained=pretrained/imagenet_mobilenetv2-1.0_best.pth.tar \
--save=./result/imagenet/mobilenetv2 \
--iters=200 \
--log_name='imagenet_mobilenet_param_1_flops_0.7_iter200' \
--param_prune_rate=1 \
--FLOPs_prune_rate=0.7 \
--seed=0
