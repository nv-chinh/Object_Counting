#!/bin/sh
export CUDA_VISIBLE_DEVICES=7  # Set the GPU ID. Comment this line to use all available GPUs.

### Some notes:
# 1. The training script will automatically use all available GPUs in the DDP mode.
# 2. You can use the `--amp` argument to enable automatic mixed precision training to speed up the training process. Could be useful for UCF-QNRF and NWPU.
# 3. Valid values for `--dataset` are `nwpu`, `sha`, `shb`, and `qnrf`.
# See the `trainer.py` for more details.

# # Train the commonly used VGG19-based encoder-decoder model on NWPU-Crowd.
# python trainer.py \
#     --model vgg19_ae --input_size 448 --reduction 8 --truncation 4 --anchor_points average \
#     --dataset nwpu \
#     --count_loss dmcount &&

# Train the CLIP-EBC (ResNet50) model on ShanghaiTech A. Use `--dataset shb` if you want to train on ShanghaiTech B.
# python trainer.py \
#     --model clip_resnet50 --input_size 448 --reduction 32 --truncation 19 --anchor_points average --prompt_type word \
#     --dataset jhu \
#     --count_loss dmcount

python trainer.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --dataset shb \
    --count_loss dmcount

# # Train the CLIP-EBC (ViT-B/16) model on UCF-QNRF, using VPT in training and sliding window prediction in testing.
# # By default, 32 tokens for each layer are used in VPT. You can also set `--num_vpt` to change the number of tokens.
# # By default, the deep visual prompt tuning is used. You can set `--shallow_vpt` to use the shallow visual prompt tuning.
# python trainer.py \
#     --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 \
#     --dataset qnrf --batch_size 16 --amp \
#     --num_crops 2 --sliding_window --window_size 224 --stride 224 --warmup_lr 1e-3 \
#     --count_loss dmcount
# python trainer.py \
#     --model clip_vit_b_16 --input_size 448 --reduction 16 --truncation 8 \
#     --dataset jhu --batch_size 8 --amp \
#     --num_crops 2 --sliding_window --window_size 448 --stride 448 --warmup_lr 1e-3 \
#     --count_loss dmcount