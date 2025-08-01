import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from argparse import ArgumentParser
import os, json
os.environ["CUDA_VISIBLE_DEVICES"]="2"
current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log
from train import train
from eval import evaluate
from tqdm import tqdm
import numpy as np
import cv2
import shutil
import torch.nn.functional as F
import math

parser = ArgumentParser(description="Train an EBC model.")

# Parameters for model
parser.add_argument("--model", type=str, default="clip_resnet50", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=112, choices=[8, 16, 32, 64, 112, 224], help="The reduction factor of the model.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
parser.add_argument("--truncation", type=int, default=19, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The prompt type for CLIP.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"], help="The granularity of bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")

# Parameters for dataset
parser.add_argument("--data_path", type=str, default = "/home/chinhbrian/CLIP-EBC/Cattle_Counting_Dataset", help="The dataset to train on.")
parser.add_argument("--output_path", type=str, default = "results_test", help="The dataset to train on.")
parser.add_argument("--dataset", type=str, default = "jhu", help="The dataset to train on.")
parser.add_argument("--batch_size", type=int, default=8, help="The training batch size.")
parser.add_argument("--num_crops", type=int, default=1, help="The number of crops for multi-crop training.")
parser.add_argument("--min_scale", type=float, default=1.0, help="The minimum scale for random scale augmentation.")
parser.add_argument("--max_scale", type=float, default=2.0, help="The maximum scale for random scale augmentation.")
parser.add_argument("--brightness", type=float, default=0.1, help="The brightness factor for random color jitter augmentation.")
parser.add_argument("--contrast", type=float, default=0.1, help="The contrast factor for random color jitter augmentation.")
parser.add_argument("--saturation", type=float, default=0.1, help="The saturation factor for random color jitter augmentation.")
parser.add_argument("--hue", type=float, default=0.0, help="The hue factor for random color jitter augmentation.")
parser.add_argument("--kernel_size", type=int, default=5, help="The kernel size for Gaussian blur augmentation.")
parser.add_argument("--saltiness", type=float, default=1e-3, help="The saltiness for pepper salt noise augmentation.")
parser.add_argument("--spiciness", type=float, default=1e-3, help="The spiciness for pepper salt noise augmentation.")
parser.add_argument("--jitter_prob", type=float, default=0.2, help="The probability for random color jitter augmentation.")
parser.add_argument("--blur_prob", type=float, default=0.2, help="The probability for Gaussian blur augmentation.")
parser.add_argument("--noise_prob", type=float, default=0.5, help="The probability for pepper salt noise augmentation.")

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--window_size", type=int, default=None, help="The window size for in prediction.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to the nearest multiple of the input size.")

# Parameters for loss function
parser.add_argument("--weight_count_loss", type=float, default=1.0, help="The weight for count loss.")
parser.add_argument("--count_loss", type=str, default="dmcount", choices=["mae", "mse", "dmcount"], help="The loss function for count.")

# Parameters for optimizer (Adam)
parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay.")

# Parameters for learning rate scheduler
parser.add_argument("--warmup_epochs", type=int, default=50, help="Number of epochs for warmup. The learning rate will increase from eta_min to lr.")
parser.add_argument("--warmup_lr", type=float, default=1e-6, help="Learning rate for warmup.")
parser.add_argument("--T_0", type=int, default=5, help="Number of epochs for the first restart.")
parser.add_argument("--T_mult", type=int, default=2, help="A factor increases T_0 after a restart.")
parser.add_argument("--eta_min", type=float, default=1e-7, help="Minimum learning rate.")

# Parameters for training
parser.add_argument("--total_epochs", type=int, default=2600, help="Number of epochs to train.")
parser.add_argument("--eval_start", type=int, default=50, help="Start to evaluate after this number of epochs.")
parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate every this number of epochs.")
parser.add_argument("--save_freq", type=int, default=5, help="Save checkpoint every this number of epochs. Could help reduce I/O.")
parser.add_argument("--save_best_k", type=int, default=3, help="Save the best k checkpoints.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--weight_path", type=str, default='/home/chinhbrian/CLIP-EBC/checkpoints/jhu/clip_resnet50_word_448_112_19_fine_1.0_dmcount/best_mae_0.pth', help="The path to the weights of the model.")

def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    print(f"Initialized successfully. Training with {nprocs} GPUs.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:7"
    print(f"Using device: {device}.")

    ddp = nprocs > 1

    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)][args.dataset]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    args.bins = bins
    args.anchor_points = anchor_points

    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    )

    state_dict = torch.load(args.weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(args.weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    val_loader = get_dataloader(args, split="test", ddp=False, mode = 'inference')

    list_err = []
    list_gt = []
    for image, num_gt, image_name in tqdm(val_loader):
        image_name = image_name[0]
        height, width = image.shape[-2:]
        image = image.to(device)
        target_count = num_gt.item()

        with torch.set_grad_enabled(False):
            pred_density = model(image)
            pred_count = int(pred_density.sum(dim=(1, 2, 3)).cpu().numpy().tolist()[0])
            err = abs(pred_count-target_count)
            print(image_name, target_count, pred_count, err)
            list_err.append(err)
            list_gt.append(target_count)

            pred_density_upsampled = F.interpolate(
                    pred_density,
                    size=(height, width),  # (H, W)
                    mode='bilinear',
                    align_corners=False
                )
            vis_img = pred_density_upsampled[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

            # Overlay text with GT, Pred, Acc
            text_1 = f"GT: {target_count}"
            text_2 = f"Pred: {pred_count}"
            text_3 = f"Err: {round(err, 4)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 0, 255)
            thickness = 2
            # cv2.putText(vis_img, text_1, (10, 25), font, font_scale, color, thickness)
            # cv2.putText(vis_img, text_2, (10, 50), font, font_scale, color, thickness)
            # cv2.putText(vis_img, text_3, (10, 75), font, font_scale, color, thickness)
            cv2.putText(vis_img, text_1, (10, 25), font, font_scale, color, thickness)
            cv2.putText(vis_img, text_2, (10, 65), font, font_scale, color, thickness)
            cv2.putText(vis_img, text_3, (10, 105), font, font_scale, color, thickness)
            

            cv2.imwrite(os.path.join(args.output_path, image_name[:-4] + '_ovl.bmp'), vis_img)
            shutil.copyfile(os.path.join(args.data_path, 'test', 'images', image_name), os.path.join(args.output_path, image_name))
    mae = sum(list_err) / len(list_err)
    avg_err = 1 - sum(list_err) / sum(list_gt)
    squared_err = [x**2 for x in list_err]
    rmse = math.sqrt(sum(squared_err)/len(squared_err))

    print(f"MAE: {round(mae, 4)}, RMSE: {round(rmse, 4)}, Accuracy: {round(avg_err,4)}")

def main():
    args = parser.parse_args()
    args.model = args.model.lower()
    args.dataset = standardize_dataset_name(args.dataset)

    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None
    
    if "clip" not in args.model:
        args.prompt_type = None

    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."

    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
