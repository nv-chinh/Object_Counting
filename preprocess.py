import os
import glob
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union, Optional
from warnings import warn


def _calc_size(
    img_w: int,
    img_h: int,
    min_size: int,
    max_size: int,
    base: int = 32
) -> Union[Tuple[int, int], None]:
    """
    This function generates a new size for an image while keeping the aspect ratio. The new size should be within the given range (min_size, max_size).

    Args:
        img_w (int): The width of the image.
        img_h (int): The height of the image.
        min_size (int): The minimum size of the edges of the image.
        max_size (int): The maximum size of the edges of the image.
    """
    assert min_size % base == 0, f"min_size ({min_size}) must be a multiple of {base}"
    if max_size != float("inf"):
        assert max_size % base == 0, f"max_size ({max_size}) must be a multiple of {base} if provided"

    assert min_size <= max_size, f"min_size ({min_size}) must be less than or equal to max_size ({max_size})"

    aspect_ratios = (img_w / img_h, img_h / img_w)
    if min_size / max_size <= min(aspect_ratios) <= max(aspect_ratios) <= max_size / min_size:  # possible to resize and preserve the aspect ratio
        if min_size <= min(img_w, img_h) <= max(img_w, img_h) <= max_size:  # already within the range, no need to resize
            ratio = 1.
        elif min(img_w, img_h) < min_size:  # smaller than the minimum size, resize to the minimum size
            ratio = min_size / min(img_w, img_h)
        else:  # larger than the maximum size, resize to the maximum size
            ratio = max_size / max(img_w, img_h)

        new_w, new_h = int(round(img_w * ratio / base) * base), int(round(img_h * ratio / base) * base)
        new_w = max(min_size, min(max_size, new_w))
        new_h = max(min_size, min(max_size, new_h))
        return new_w, new_h

    else:  # impossible to resize and preserve the aspect ratio
        msg = f"Impossible to resize {img_w}x{img_h} image while preserving the aspect ratio to a size within the range ({min_size}, {max_size}). Will not limit the maximum size."
        warn(msg)
        return _calc_size(img_w, img_h, min_size, float("inf"), base)


def _generate_random_indices(
    total_size: int,
    out_dir: str,
) -> None:
    """
    Generate randomly selected indices for labelled data in semi-supervised learning.
    """
    rng = np.random.default_rng(42)
    for percent in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        num_select = int(total_size * percent)
        selected = rng.choice(total_size, num_select, replace=False)
        selected.sort()
        selected = selected.tolist()
        with open(os.path.join(out_dir, f"{int(percent * 100)}%.txt"), "w") as f:
            for i in selected:
                f.write(f"{i}\n")


def _resize(image: np.ndarray, label: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    image_h, image_w, _ = image.shape
    new_size = _calc_size(image_w, image_h, min_size, max_size)
    if new_size is None:
        return image, label, False
    else:
        new_w, new_h = new_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC) if (new_w, new_h) != (image_w, image_h) else image
        label = label * np.array([[new_w / image_w, new_h / image_h]]) if len(label) > 0 and (new_w, new_h) != (image_w, image_h) else label
        return image, label, True


def _preprocess(
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int) -> None:
    assert os.path.isdir(data_src_dir), f"{data_src_dir} does not exist"
    os.makedirs(data_dst_dir, exist_ok=True)
    print(f"Pre-processing dataset...")
    for split in ["train", "val"]:
        image_dir = os.path.join(data_src_dir, split, 'images')
        image_src_paths = glob.glob(os.path.join(image_dir, '*.*'))
        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        size = len(str(len(image_src_paths)))
        for i, image_src_path in tqdm(enumerate(image_src_paths), total=len(image_src_paths)):
            name = f"{(i + 1):0{size}d}"
            ext = os.path.splitext(image_src_path)[1]
            image = cv2.imread(image_src_path)
            height, width, _ = image.shape
            label_src_path = image_src_path.replace('images','labels').replace(ext,'.txt')
            with open(label_src_path, "r") as f:
                labels = f.read().splitlines()
            points = []
            for label in labels:
                _,x,y,_,_ = list(map(float, label.split(' ')))
                points.append([x * width,y * height])
            points = np.array(points)
            _resize_and_save(
                image=image,
                label=points,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                min_size=min_size,
                max_size=max_size,
                ext = ext
            )

        if split == "train":
            _generate_random_indices(len(image_src_paths), os.path.join(data_dst_dir, split))


def _resize_and_save(
    image: np.ndarray,
    name: str,
    image_dst_dir: str,
    label: Optional[np.ndarray] = None,
    label_dst_dir: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    ext: str = '.jpg'
) -> None:
    os.makedirs(image_dst_dir, exist_ok=True)

    if label is not None:
        assert label_dst_dir is not None, "label_dst_dir must be provided if label is provided"
        os.makedirs(label_dst_dir, exist_ok=True)

    image_dst_path = os.path.join(image_dst_dir, name + ext)

    if label is not None:
        label_dst_path = os.path.join(label_dst_dir, f"{name}.npy")
    else:
        label = np.array([])
        label_dst_path = None

    if min_size is not None:
        assert max_size is not None, f"max_size must be provided if min_size is provided, got {max_size}"
        image, label, success = _resize(image, label, min_size, max_size)
        if not success:
            print(f"image: {image_dst_path} is not resized")

    cv2.imwrite(image_dst_path, image)

    if label_dst_path is not None:
        np.save(label_dst_path, label)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="The root directory of the source dataset.")
    parser.add_argument("--dst_dir", type=str, required=True, help="The root directory of the destination dataset.")
    parser.add_argument("--min_size", type=int, default=448, help="The minimum size of the shorter side of the image.")
    parser.add_argument("--max_size", type=int, default=2048, help="The maximum size of the longer side of the image.")

    args = parser.parse_args()
    args.src_dir = os.path.abspath(args.src_dir)
    args.dst_dir = os.path.abspath(args.dst_dir)
    args.max_size = float("inf") if args.max_size is None else args.max_size
    return args


if __name__ == "__main__":
    args = parse_args()
    _preprocess(
        data_src_dir=args.src_dir,
        data_dst_dir=args.dst_dir,
        min_size=args.min_size,
        max_size=args.max_size
    )
