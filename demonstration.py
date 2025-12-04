from typing import List
import torch
import argparse
from dataclasses import dataclass
from models import ResNet, Generator, GeneratorConfig, ResNetConfig
from PIL import Image, ImageDraw, ImageFont

from utils.transforms import convert_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Args:
    resnet: str
    gan: str
    image: str
    out: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Demonstrate Super-Resolution")

    parser.add_argument(
        "--resnet", type=str, help="Path to ResNet model", required=True
    )
    parser.add_argument("--gan", type=str, help="Path to GAN model", required=True)
    parser.add_argument("--image", type=str, help="Input image", required=True)
    parser.add_argument("--out", type=str, help="Output image", required=True)

    args = parser.parse_args()

    args = Args(
        resnet=args.resnet,
        gan=args.gan,
        image=args.image,
        out=args.out,
    )

    return args


def make_labeled_2x2_grid(
    images: List[Image.Image],
    labels: List[str],
    title="Super Resolution",
    cell_size=(200, 200),
):
    assert len(images) == 4
    assert len(labels) == 4

    images = [img.resize(cell_size) for img in images]

    grid_w = cell_size[0] * 2
    grid_h = cell_size[1] * 2

    title_h = 50
    label_h = 30

    total_h = title_h + grid_h + 2 * label_h
    canvas = Image.new("RGB", (grid_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_label = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((grid_w - title_w) // 2, 10), title, fill="black", font=font_title)

    base_y = title_h
    positions = [
        (0, base_y),
        (cell_size[0], base_y),
        (0, base_y + cell_size[1] + label_h),
        (cell_size[0], base_y + cell_size[1] + label_h),
    ]

    for img, label, pos in zip(images, labels, positions):
        x, y = pos
        canvas.paste(img, pos)

        label_y = y + cell_size[1] + 5
        label_bbox = draw.textbbox((0, 0), label, font=font_label)
        label_w = label_bbox[2] - label_bbox[0]
        draw.text(
            (x + (cell_size[0] - label_w) // 2, label_y),
            label,
            fill="black",
            font=font_label,
        )

    return canvas


def main(args: Args):
    print("Loading image...")
    hr = Image.open(args.image)
    assert hr.width <= 2048, "Image too wide"
    assert hr.height <= 2048, "Image too tall"

    hr = hr.convert("RGB")

    lr = hr.resize(
        (
            int(hr.width / 4),
            int(hr.height / 4),
        ),
        Image.Resampling.BICUBIC,
    )

    lr_upsampled = lr.resize((hr.width, hr.height), Image.Resampling.BICUBIC)
    lr = convert_image(lr, "pil", "[-1, 1]")

    assert not isinstance(lr, Image.Image)

    print("Loading models...")
    resnet_state_dict = torch.load(args.resnet, weights_only=False)
    resnet_config_dict = resnet_state_dict.get("config", {})
    resnet_config = ResNetConfig(**resnet_config_dict)
    resnet = ResNet(resnet_config)
    resnet.load_state_dict(resnet_state_dict["model"])
    resnet = resnet.to(device)

    generator_state_dict = torch.load(args.gan, weights_only=False)
    generator_config_dict = generator_state_dict.get("generator_config", {})
    generator_config = GeneratorConfig(**generator_config_dict)
    generator = Generator(generator_config)
    generator.load_state_dict(torch.load(args.gan, map_location=device)["generator"])
    generator = generator.to(device)

    resnet.eval()
    generator.eval()

    for n, p in resnet.named_parameters():
        if p.dtype != torch.float32:
            print("[ResNet]", n, p.dtype)

    for n, p in generator.named_parameters():
        if p.dtype != torch.float32:
            print("[GAN]", n, p.dtype)

    print("Running inference...")
    with torch.no_grad():
        resnet_sr = resnet(lr.unsqueeze(0).to(device)).squeeze(0)
        print("ResNet complete...")
        generator_sr = generator(lr.unsqueeze(0).to(device)).squeeze(0)
        print("GAN complete...")

    resnet_sr = convert_image(resnet_sr.clamp(-1, 1), "[-1, 1]", "pil")
    generator_sr = convert_image(generator_sr.clamp(-1, 1), "[-1, 1]", "pil")
    assert not isinstance(resnet_sr, torch.Tensor)
    assert not isinstance(generator_sr, torch.Tensor)

    print("Generating output...")
    result = make_labeled_2x2_grid(
        [lr_upsampled, hr, resnet_sr, generator_sr],
        ["Low Resolution", "High Resolution", "SRResNet", "SRGAN"],
        cell_size=hr.size,
    )

    result.save(args.out)
    print(f"Done! Saved to {args.out}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
