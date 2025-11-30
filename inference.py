import torch
import argparse
from dataclasses import dataclass
from models import ResNet, ResNetConfig, Generator, GeneratorConfig
from PIL import Image

from utils import convert_image


@dataclass
class Args:
    model: str
    input: str
    output: str
    tile: int = 256  # patch size
    tile_overlap: int = 16  # overlap size


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Run SR inference with ResNet")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    return Args(**vars(parser.parse_args()))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tile_forward(model, img, tile, overlap, scale):
    b, c, h, w = img.size()
    output = torch.zeros(b, c, h * scale, w * scale).to(device)

    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            y1, x1 = y, x
            y2 = min(y1 + tile, h)
            x2 = min(x1 + tile, w)

            in_patch = img[:, :, y1:y2, x1:x2]
            sr_patch = model(in_patch.to(device))

            dy1, dx1 = y1 * scale, x1 * scale
            dy2, dx2 = y2 * scale, x2 * scale

            output[:, :, dy1:dy2, dx1:dx2] = sr_patch

    return output


def main(args: Args):
    print("Loading image...")
    image = Image.open(args.input).convert("RGB")
    tensor = convert_image(image, "pil", "[-1, 1]")
    assert not isinstance(tensor, Image.Image)
    tensor = tensor.unsqueeze(0).to(device)

    print("Loading model...")

    checkpoint = torch.load(args.model, weights_only=False)
    if checkpoint.get("model"):
        config_dict = checkpoint.get("config", {})
        config = ResNetConfig(**config_dict)
        weights = checkpoint["model"]
        model = ResNet(config)
        model.load_state_dict(weights)
    else:
        config_dict = checkpoint.get("generator_config", {})
        config = GeneratorConfig(**config_dict)
        weights = checkpoint["generator"]
        model = Generator(config)
        model.load_state_dict(weights)

    model = model.to(device)
    model.eval()

    scale = 4

    print("Running tiled inference...")
    with torch.no_grad():
        out_tensor = tile_forward(
            model,
            tensor,
            tile=args.tile,
            overlap=args.tile_overlap,
            scale=scale,
        )
    print("Inference complete!")

    out_tensor = out_tensor.squeeze(0).cpu().clamp(-1, 1)
    result = convert_image(out_tensor, "[-1, 1]", "pil")
    assert not isinstance(result, torch.Tensor)
    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main(parse_args())
