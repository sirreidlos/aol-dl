from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch.backends.cudnn as cudnn
import torch
from torch import Tensor, nn, optim
from models.srgan import Generator, GeneratorConfig, Discriminator, DiscriminatorConfig
from models import TruncatedVGG19
from utils.dataset import SRDataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import os


import ssl
import certifi

from utils.transforms import convert_image

ssl._create_default_https_context = ssl._create_default_https_context = (
    lambda: ssl.create_default_context(cafile=certifi.where())
)


@dataclass
class Args:
    checkpoints_dir: str = "./checkpoints"
    warmup_model: str | None = None
    data_folder: str = "./"
    crop_size: int = 96
    convolutional_block_count: int = 7
    scaling_factor: int = 4
    dense_size: int = 1024
    large_kernel_size: int = 9
    kernel_size: int = 3
    channels: int = 64
    blocks: int = 16
    checkpoint: str | None = None
    batch_size: int = 16
    start_epoch: int = 0
    iterations: float = 1e6
    workers: int = 4
    lr: float = 1e-4
    discriminator_lr: float = 1e-4
    grad_clip: float | None = None
    k: int = 1
    label_smoothing: float = 0.1
    vgg_layer: Tuple[int, int] = (5, 4)
    loss_strategy: str = "perceptual"


def parse_args() -> Args:
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
        "--warmup_model",
        type=str,
        help="Path to a warmed-up model checkpoint",
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
        "--discriminator_lr",
        type=float,
        default=None,
        help="Discriminator learning rate",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Gradient clipping threshold (None to disable)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of generator updates per discriminator update",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--vgg_layer", nargs=2, type=int, default=[5, 4])
    parser.add_argument(
        "--loss_Strategy",
        type=str,
        choices=["perceptual", "content"],
        default="perceptual",
    )

    args = parser.parse_args()

    if args.discriminator_lr is None:
        args.discriminator_lr = args.lr

    args.vgg_layer = tuple(args.vgg_layer)

    return Args(**vars(args))


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class CheckpointManager:
    def __init__(self, checkpoints_dir: str):
        self.path = Path(checkpoints_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        generator: Generator,
        discriminator: Discriminator,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        loss: Optional[Tuple[float, float]] = None,
    ) -> None:
        filename = f"srgan_{epoch + 1}.pth.tar"

        # Validation every N epochs
        # if (epoch + 1) % 5 == 0:
        #     val_psnr, val_ssim = 0.0, 0.0
        #     # TODO: proper PSNR & SSIM validation
        #     print(
        #         f"Validation — Epoch {epoch + 1}: PSNR={{val_psnr:.2f}}, SSIM={{val_ssim:.3f}}"
        #     )

        save_object = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
        }

        if loss is not None:
            save_object["loss"] = loss

        torch.save(
            save_object,
            os.path.join(self.path, filename),
        )

    def load(
        self,
        checkpoint_path: str,
        generator: Generator,
        discriminator: Discriminator,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        device: Optional[torch.device] = None,
    ) -> int:
        if device is None:
            device = DEFAULT_DEVICE
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

        return checkpoint.get("epoch")


class SRGANLossStrategy(ABC):
    @abstractmethod
    def calculate_loss_g(self, *args, **kwargs) -> Tensor: ...

    @abstractmethod
    def calculate_loss_d(self, *args, **kwargs) -> Tensor: ...


class SRGANContentLossStrategy(SRGANLossStrategy):
    def __init__(
        self,
        adversarial_loss_weight: float,
        device=DEFAULT_DEVICE,
        label_smoothing: float = 0.0,
    ):
        self.adversarial_loss_weight = adversarial_loss_weight
        self.label_smoothing = label_smoothing

        self.content_loss_criterion = nn.MSELoss().to(device)
        self.adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    def calculate_loss_g(
        self,
        generator: Generator,
        discriminator: Discriminator,
        lr_imgs: Tensor,
        hr_imgs: Tensor,
    ) -> Tensor:
        """
        lr_imgs: [-1, 1]
        hr_imgs: [-1, 1]
        """
        sr_imgs = generator(lr_imgs).clamp(-1, 1)
        hr_imgs = hr_imgs.detach()

        discriminator_prediction = discriminator(sr_imgs)
        content_loss = self.content_loss_criterion(sr_imgs, hr_imgs)
        adversarial_loss = self.adversarial_loss_criterion(
            discriminator_prediction,
            torch.ones_like(discriminator_prediction),
        )
        perceptual_loss = content_loss + self.adversarial_loss_weight * adversarial_loss

        return perceptual_loss

    def calculate_loss_d(
        self,
        discriminator: Discriminator,
        sr_imgs: Tensor,
        hr_imgs: Tensor,
    ) -> Tensor:
        discriminator_sr_prediction = discriminator(sr_imgs)
        discriminator_hr_prediction = discriminator(hr_imgs)

        fake_targets = torch.zeros_like(discriminator_sr_prediction)
        real_targets = torch.ones_like(discriminator_hr_prediction) * (
            1.0 - self.label_smoothing
        )

        discriminator_loss = self.adversarial_loss_criterion(
            discriminator_sr_prediction, fake_targets
        ) + self.adversarial_loss_criterion(discriminator_hr_prediction, real_targets)

        return discriminator_loss


class SRGANPerceptualLossStrategy(SRGANLossStrategy):
    def __init__(
        self,
        adversarial_loss_weight: float,
        device=DEFAULT_DEVICE,
        label_smoothing: float = 0.0,
        vgg_layer=(5, 4),
    ):
        self.adversarial_loss_weight = adversarial_loss_weight
        self.vgg = TruncatedVGG19(selected=vgg_layer).to(DEFAULT_DEVICE)
        self.label_smoothing = label_smoothing

        self.content_loss_criterion = nn.MSELoss().to(device)
        self.adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    def calculate_loss_g(
        self,
        generator: Generator,
        discriminator: Discriminator,
        lr_imgs: Tensor,
        hr_imgs: Tensor,
    ) -> Tensor:
        """
        lr_imgs: [-1, 1]
        hr_imgs: [-1, 1]
        """
        sr_imgs = generator(lr_imgs).clamp(-1, 1)

        hr_imgs_imgnet = convert_image(hr_imgs, "[-1, 1]", "imagenet-norm")
        sr_imgs_imgnet = convert_image(sr_imgs, "[-1, 1]", "imagenet-norm")

        vgg_hr_imgs = self.vgg(hr_imgs_imgnet).detach()
        vgg_sr_imgs = self.vgg(sr_imgs_imgnet)

        discriminator_prediction = discriminator(sr_imgs)
        content_loss = self.content_loss_criterion(vgg_sr_imgs, vgg_hr_imgs)
        adversarial_loss = self.adversarial_loss_criterion(
            discriminator_prediction,
            torch.ones_like(discriminator_prediction),
        )
        perceptual_loss = content_loss + self.adversarial_loss_weight * adversarial_loss

        return perceptual_loss

    def calculate_loss_d(
        self,
        discriminator: Discriminator,
        sr_imgs: Tensor,
        hr_imgs: Tensor,
    ) -> Tensor:
        discriminator_sr_prediction = discriminator(sr_imgs)
        discriminator_hr_prediction = discriminator(hr_imgs)

        fake_targets = torch.zeros_like(discriminator_sr_prediction)
        real_targets = torch.ones_like(discriminator_hr_prediction) * (
            1.0 - self.label_smoothing
        )

        discriminator_loss = self.adversarial_loss_criterion(
            discriminator_sr_prediction, fake_targets
        ) + self.adversarial_loss_criterion(discriminator_hr_prediction, real_targets)

        return discriminator_loss


class SRGANTrainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        loss_strategy: SRGANLossStrategy,
        train_loader: DataLoader,
        checkpoint_manager: CheckpointManager,
        device: torch.device = DEFAULT_DEVICE,
        grad_clip: Optional[float] = None,
        k: int = 1,
        label_smoothing: float = 0.0,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_strategy = loss_strategy
        self.train_loader = train_loader
        self.device = device
        self.grad_clip = grad_clip
        self.k = k
        self.checkpoint_manager = checkpoint_manager
        self.label_smoothing = label_smoothing

    def train(self, start_epoch: int, iterations: float) -> List[Tuple[float, float]]:
        loss_history = []

        epochs = int(iterations // len(self.train_loader) + 1)
        pbar = tqdm(
            range(start_epoch, epochs),
            desc="Epochs",
            ncols=100,
            position=0,
            initial=start_epoch,
            total=epochs,
        )
        for epoch in pbar:
            total_g_loss, total_d_loss = self.train_epoch(epoch)
            pbar.set_postfix(
                {
                    "TGLoss": f"{total_g_loss:.4f}",
                    "TDLoss": f"{total_d_loss:.4f}",
                }
            )

            loss_history.append([total_g_loss, total_d_loss])

            self.checkpoint_manager.save(
                epoch,
                self.generator,
                self.discriminator,
                self.generator_optimizer,
                self.discriminator_optimizer,
                loss=(total_g_loss, total_d_loss),
            )

        return loss_history

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        batch_pbar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
            ncols=100,
            position=1,
            leave=False,
        )

        for batch in batch_pbar:
            g_loss, d_loss = self.train_step(batch)
            total_generator_loss += g_loss
            total_discriminator_loss += d_loss
            batch_pbar.set_postfix({"GLoss": f"{g_loss:.4f}", "DLoss": f"{d_loss:.4f}"})

        return total_generator_loss / len(
            self.train_loader
        ), total_discriminator_loss / len(self.train_loader)

    def train_step(self, batch) -> Tuple[float, float]:
        self.generator.train()
        self.discriminator.train()

        lr_imgs, hr_imgs = batch
        lr_imgs = lr_imgs.to(self.device)
        hr_imgs = hr_imgs.to(self.device)

        total_perceptual_loss = 0.0

        for _ in range(self.k):
            perceptual_loss = self.loss_strategy.calculate_loss_g(
                self.generator,
                self.discriminator,
                lr_imgs,
                hr_imgs,
            )

            self.generator_optimizer.zero_grad()
            perceptual_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_value_(self.generator.parameters(), self.grad_clip)
            self.generator_optimizer.step()

            total_perceptual_loss += perceptual_loss.item()

        with torch.no_grad():
            sr_imgs = self.generator(lr_imgs).detach().clamp(-1, 1)

        discriminator_loss = self.loss_strategy.calculate_loss_d(
            self.discriminator,
            sr_imgs,
            hr_imgs,
        )

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_value_(self.discriminator.parameters(), self.grad_clip)
        self.discriminator_optimizer.step()

        del lr_imgs, hr_imgs, sr_imgs

        return total_perceptual_loss / max(1, self.k), discriminator_loss.item()


def main():
    args = parse_args()
    device = DEFAULT_DEVICE

    generator_config = GeneratorConfig(
        residual_block_count=args.blocks,
        scaling_factor=args.scaling_factor,
        large_kernel_size=args.large_kernel_size,
        small_kernel_size=args.kernel_size,
        channels=args.channels,
    )
    generator = Generator(generator_config)
    generator = generator.to(DEFAULT_DEVICE)
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.lr)

    discriminator_config = DiscriminatorConfig(
        convolutional_block_count=args.convolutional_block_count,
        kernel_size=args.kernel_size,
        channels=args.channels,
        dense_size=args.dense_size,
    )
    discriminator = Discriminator(discriminator_config)
    discriminator = discriminator.to(DEFAULT_DEVICE)
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(), lr=args.lr
    )

    start_epoch = args.start_epoch

    if args.warmup_model is not None:
        checkpoint = torch.load(args.warmup_model)
        generator.resnet.load_state_dict(checkpoint["model"])
        generator_optimizer.load_state_dict(checkpoint["optimizer"])

    checkpoint_manager = CheckpointManager(args.checkpoints_dir)
    if args.checkpoint is not None:
        cp_epoch = checkpoint_manager.load(
            args.checkpoint,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            device=device,
        )
        start_epoch = cp_epoch + 1

    match args.loss_strategy:
        case "perceptual":
            loss_strategy = SRGANPerceptualLossStrategy(
                5e-3, device, args.label_smoothing, vgg_layer=args.vgg_layer
            )
        case "content":
            loss_strategy = SRGANContentLossStrategy(5e-3, device, args.label_smoothing)

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

    trainer = SRGANTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_strategy=loss_strategy,
        train_loader=train_loader,
        checkpoint_manager=checkpoint_manager,
        device=device,
        grad_clip=args.grad_clip,
        k=args.k,
        label_smoothing=args.label_smoothing,
    )

    loss_history = trainer.train(start_epoch, args.iterations)


if __name__ == "__main__":
    main()
