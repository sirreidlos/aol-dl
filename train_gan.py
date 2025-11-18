from typing import Tuple
import torch.backends.cudnn as cudnn
import torch
from torch import Tensor, nn, optim
from models import (
    Generator,
    GeneratorConfig,
    Discriminator,
    DiscriminatorConfig,
    TruncatedVGG19,
)
from utils.dataset import SRDataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import argparse


import ssl
import certifi

from utils.transforms import convert_image

ssl._create_default_https_context = ssl._create_default_https_context = (
    lambda: ssl.create_default_context(cafile=certifi.where())
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Super-Resolution Generative Adversarial Network Model"
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
        "--convolutional_block_count",
        type=int,
        default=7,
        help="Number of convolutional blocks in the discriminator",
    )
    parser.add_argument(
        "--scaling_factor",
        type=int,
        default=4,
        help="Scaling factor for the generator (LR images will be downsampled from HR images by this factor)",
    )
    parser.add_argument(
        "--dense_size",
        type=int,
        default=1024,
        help="Size of dense layer in the discriminator",
    )
    parser.add_argument(
        "--large_kernel_size",
        type=int,
        default=9,
        help="Kernel size of the first and last convolutions",
    )
    parser.add_argument(
        "--kernel_size",
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

    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    args = parse_args()

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    generator_config = GeneratorConfig(
        residual_block_count=args.blocks,
        scaling_factor=args.scaling_factor,
        large_kernel_size=args.large_kernel_size,
        small_kernel_size=args.kernel_size,
        channels=args.channels,
    )
    generator = Generator(generator_config)
    generator_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr
    )

    discriminator_config = DiscriminatorConfig(
        convolutional_block_count=args.convolutional_block_count,
        kernel_size=args.kernel_size,
        channels=args.channels,
        dense_size=args.dense_size,
    )
    discriminator = Discriminator(discriminator_config)
    discriminator_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=args.lr,
    )

    if args.checkpoint is None:
        start_epoch = args.start_epoch

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

    vgg = TruncatedVGG19(selected=(5, 4)).to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    content_loss_criterion = nn.MSELoss().to(device)
    adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)
    adversarial_loss_weight = 10e-3

    train_dataset = SRDataset(
        args.data_folder,
        split="train",
        crop_size=args.crop_size,
        scaling_factor=args.scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="imagenet-norm",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    epochs = int(args.iterations // len(train_loader) + 1)
    pbar = tqdm(range(start_epoch, epochs), desc="Epochs", ncols=100, position=0)
    for epoch in pbar:
        total_generator_loss, total_discriminator_loss = train_epoch(
            train_loader=train_loader,
            generator=generator,
            discriminator=discriminator,
            vgg=vgg,
            content_loss_criterion=content_loss_criterion,
            adversarial_loss_criterion=adversarial_loss_criterion,
            adversarial_loss_weight=adversarial_loss_weight,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )

        pbar.set_postfix(
            {
                "TGLoss": f"{total_generator_loss:.4f}",
                "TDLoss": f"{total_discriminator_loss:.4f}",
            }
        )

        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(),
            },
            f"{args.checkpoints_dir}/srgan_{epoch}.pth.tar",
        )


def train_epoch(
    train_loader: DataLoader,
    generator: Generator,
    discriminator: Discriminator,
    vgg: TruncatedVGG19,
    content_loss_criterion: nn.MSELoss,
    adversarial_loss_criterion: nn.BCEWithLogitsLoss,
    adversarial_loss_weight: float,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    epoch: int,
    grad_clip: float | None,
):
    total_generator_loss = 0.0
    total_discriminator_loss = 0.0

    batch_pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        ncols=100,
        position=1,
        leave=False,
    )

    for batch in batch_pbar:
        g_loss, d_loss = train_step(
            generator=generator,
            discriminator=discriminator,
            vgg=vgg,
            batch=batch,
            content_loss_criterion=content_loss_criterion,
            adversarial_loss_criterion=adversarial_loss_criterion,
            adversarial_loss_weight=adversarial_loss_weight,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            grad_clip=grad_clip,
        )

        total_generator_loss += g_loss
        total_discriminator_loss += d_loss

        batch_pbar.set_postfix(
            {
                "GLoss": f"{g_loss:.4f}",
                "DLoss": f"{d_loss:.4f}",
            }
        )

    return (
        total_generator_loss / len(train_loader),
        total_discriminator_loss / len(train_loader),
    )


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    vgg: TruncatedVGG19,
    batch: Tuple[Tensor, Tensor],
    content_loss_criterion: nn.MSELoss,
    adversarial_loss_criterion: nn.BCEWithLogitsLoss,
    adversarial_loss_weight: float,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    grad_clip: float | None,
):
    generator.train()
    discriminator.train()

    lr_imgs, hr_imgs = batch
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)
    sr_imgs = generator(lr_imgs)
    sr_imgs = convert_image(sr_imgs, "[-1, 1]", "imagenet-norm")

    vgg_hr_imgs = vgg(hr_imgs).detach()
    vgg_sr_imgs = vgg(sr_imgs)

    discriminator_prediction = discriminator(sr_imgs)
    content_loss = content_loss_criterion(vgg_sr_imgs, vgg_hr_imgs)
    adversarial_loss = adversarial_loss_criterion(
        discriminator_prediction, torch.ones_like(discriminator_prediction)
    )
    perceptual_loss = content_loss + adversarial_loss_weight * adversarial_loss

    generator_optimizer.zero_grad()
    perceptual_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_value_(generator.parameters(), grad_clip)
    generator_optimizer.step()

    sr_imgs = sr_imgs.detach()
    discriminator_sr_prediction = discriminator(sr_imgs)
    discriminator_hr_prediction = discriminator(hr_imgs)
    discriminator_loss = adversarial_loss_criterion(
        discriminator_sr_prediction, torch.zeros_like(discriminator_sr_prediction)
    ) + adversarial_loss_criterion(
        discriminator_hr_prediction, torch.ones_like(discriminator_hr_prediction)
    )

    discriminator_optimizer.zero_grad()
    discriminator_loss.backward()

    if grad_clip is not None:
        nn.utils.clip_grad_value_(discriminator.parameters(), grad_clip)
    discriminator_optimizer.step()

    del lr_imgs, hr_imgs, sr_imgs, vgg_hr_imgs, vgg_sr_imgs
    return perceptual_loss.item(), discriminator_loss.item()


if __name__ == "__main__":
    main()
