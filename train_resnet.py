from dataclasses import dataclass
import dataclasses
from typing import Tuple
import torch.backends.cudnn as cudnn
import torch
from torch import Tensor, nn, optim
from models import ResNet, ResNetConfig
from utils.dataset import SRDataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import argparse


@dataclass
class Args:
    checkpoints_dir: str
    data_folder: str
    crop_size: int
    scaling_factor: int
    large_kernel: int
    small_kernel: int
    channels: int
    blocks: int
    checkpoint: str | None
    batch_size: int
    start_epoch: int
    iterations: float
    workers: int
    lr: float
    grad_clip: float | None
    checkpoint_prefix: str
    exclude_bn: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train a Super-Resolution ResNet Model"
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--data_folder", type=str, default="./", help="Folder with JSON data files"
    )
    parser.add_argument(
        "--crop_size", type=int, default=96, help="Crop size of target HR images"
    )
    parser.add_argument(
        "--scaling_factor",
        type=int,
        default=4,
        help="Scaling factor for the generator (LR images will be downsampled from HR images by this factor)",
    )
    parser.add_argument(
        "--large_kernel",
        type=int,
        default=9,
        help="Kernel size of the first and last convolutions",
    )
    parser.add_argument(
        "--small_kernel",
        type=int,
        default=3,
        help="Kernel size of all intermediate convolutions (residual/subpixel blocks)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of channels in the residual and subpixel convolutional blocks",
    )
    parser.add_argument(
        "--blocks", type=int, default=16, help="Number of residual blocks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (None if not loading from checkpoint)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="Epoch to start training from"
    )
    parser.add_argument(
        "--iterations", type=float, default=1e6, help="Number of training iterations"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Gradient clipping threshold (None to disable)",
    )
    parser.add_argument("--checkpoint_prefix", type=str, default="srresnet_")
    parser.add_argument("--exclude_bn", action="store_true")

    args = parser.parse_args()

    return Args(**vars(args))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    args = parse_args()
    print(args)

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    config = ResNetConfig(
        residual_block_count=args.blocks,
        scaling_factor=args.scaling_factor,
        large_kernel_size=args.large_kernel,
        small_kernel_size=args.small_kernel,
        channels=args.channels,
        include_bn=not args.exclude_bn,
    )

    model = ResNet(config).to(device)
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    start_epoch = args.start_epoch

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(
        args.data_folder,
        split="train",
        crop_size=args.crop_size,
        scaling_factor=args.scaling_factor,
        lr_img_type="[-1, 1]",
        hr_img_type="[-1, 1]",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    epochs = int(args.iterations // len(train_loader) + 1)
    pbar = tqdm(
        range(start_epoch, epochs),
        desc="Epochs",
        ncols=100,
        position=0,
        initial=start_epoch,
        total=epochs,
    )
    for epoch in pbar:
        total_loss = train_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )

        pbar.set_postfix(
            {
                "Loss": f"{total_loss:.4f}",
            }
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": total_loss,
                "config": dataclasses.asdict(config),
            },
            f"{args.checkpoints_dir}/{args.checkpoint_prefix}{epoch + 1}.pth.tar",
        )


def train_epoch(
    train_loader: DataLoader,
    model: ResNet,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    grad_clip: float | None,
):
    total_loss = 0.0

    batch_pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}",
        ncols=100,
        position=1,
        leave=False,
    )

    for i, batch in batch_pbar:
        loss = train_step(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )

        total_loss += loss

        batch_pbar.set_postfix(
            {
                "Loss": f"{(total_loss / (i + 1)):.4f}",
            }
        )

    return total_loss / len(train_loader)


def train_step(
    model: ResNet,
    batch: Tuple[Tensor, Tensor],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip: float | None,
):
    model.train()
    lr_imgs, hr_imgs = batch
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)
    sr_imgs = model(lr_imgs)
    loss = criterion(sr_imgs.clamp(-1, 1), hr_imgs)

    optimizer.zero_grad()
    loss.backward()

    if grad_clip is not None:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    optimizer.step()

    del lr_imgs, hr_imgs, sr_imgs
    return loss.item()


if __name__ == "__main__":
    main()
