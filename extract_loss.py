import argparse
from dataclasses import dataclass
import re
import os

import torch


@dataclass
class Args:
    checkpoints_dir: str
    regex: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Extract the losses stored in checkpoint files"
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--regex",
        type=str,
        help="File name regex",
        required=True,
    )

    args = parser.parse_args()
    return Args(**vars(args))


def main():
    args = parse_args()

    checkpoint_paths = []
    checkpoint_losses = {}
    base_dir = args.checkpoints_dir
    for path in os.listdir(base_dir):
        if re.match(args.regex, path) is not None:
            checkpoint_paths.append(f"{base_dir}/{path}")

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        loss = checkpoint["loss"]

        checkpoint_losses[checkpoint_path] = loss

    print(checkpoint_losses)


if __name__ == "__main__":
    main()
# "srragan_psnr_nobn\d+\.pth\.tar"
# "srragan_psnr_nobn_\d+\.pth\.tar"
