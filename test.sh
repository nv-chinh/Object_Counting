python test.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average \
    --dataset cow \
    --weight_path checkpoints/cow/best.pth \
    --data_path /home/chinhbrian/CLIP-EBC/Clean_Code/Cow_Counting_Dataset \
    --output_path results