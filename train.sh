python trainer.py \
    --model clip_resnet50 --dataset cow \
    --input_size 448 --reduction 8 --truncation 4 \
    --anchor_points average \
    --count_loss dmcount \
    --batch_size 8