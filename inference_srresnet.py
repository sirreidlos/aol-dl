# import torch
# import argparse
# from dataclasses import dataclass
# from models import ResNet, ResNetConfig
# from PIL import Image

# from utils import convert_image


# @dataclass
# class Args:
#     model: str
#     input: str
#     output: str


# def parse_args() -> Args:
#     parser = argparse.ArgumentParser(description="Run an inference with ResNet")

#     parser.add_argument("--model", type=str, help="Path to ResNet model", required=True)
#     parser.add_argument("--input", type=str, help="Input image", required=True)
#     parser.add_argument("--output", type=str, help="Output image", required=True)

#     args = parser.parse_args()

#     return Args(**vars(args))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def main(args: Args):
#     print("Loading image...")
#     input_image = Image.open(args.input)
#     input_image = input_image.convert("RGB")
#     input_image = convert_image(input_image, "pil", "imagenet-norm")
#     assert not isinstance(input_image, Image.Image)

#     print("Loading model...")
#     model = ResNet(ResNetConfig(scaling_factor=4))
#     model.load_state_dict(torch.load(args.model)["model"])
#     model = model.to(device)

#     model.eval()

#     print("Running inference...")
#     with torch.no_grad():
#         output_image = model(input_image.unsqueeze(0).to(device)).squeeze(0)
#         print("Inference complete...")

#     output_image = convert_image(output_image, "[-1, 1]", "pil")
#     assert not isinstance(output_image, torch.Tensor)

#     print("Writing output...")

#     output_image.save(args.output)
#     print(f"Done! Saved to {args.output}.")


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


import torch
import argparse
from dataclasses import dataclass
from models import ResNet, ResNetConfig
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
    tensor = convert_image(image, "pil", "imagenet-norm")
    assert not isinstance(tensor, Image.Image)
    tensor = tensor.unsqueeze(0).to(device)

    print("Loading model...")
    model = ResNet(ResNetConfig(scaling_factor=4))
    model.load_state_dict(torch.load(args.model)["model"])
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

    out_tensor = out_tensor.squeeze(0).cpu()
    result = convert_image(out_tensor, "[-1, 1]", "pil")
    assert not isinstance(result, torch.Tensor)
    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main(parse_args())
