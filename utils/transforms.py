from typing import Literal
from PIL import Image
import random
import torchvision.transforms.functional as FT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = (
    torch.FloatTensor([0.485, 0.456, 0.406])
    .to(device)
    .unsqueeze(0)
    .unsqueeze(2)
    .unsqueeze(3)
)
imagenet_std_cuda = (
    torch.FloatTensor([0.229, 0.224, 0.225])
    .to(device)
    .unsqueeze(0)
    .unsqueeze(2)
    .unsqueeze(3)
)

ValidSource = Literal["pil", "[0, 255]", "[0, 1]", "[-1, 1]"]
ValidTarget = Literal[
    "pil",
    "[0, 255]",
    "[0, 1]",
    "[-1, 1]",
    "imagenet-norm",
    "y-channel",
]


def convert_image(
    img: Image.Image | torch.Tensor, source: ValidSource, target: ValidTarget
) -> Image.Image | torch.Tensor:
    assert source in {
        "pil",
        "[0, 255]",
        "[0, 1]",
        "[-1, 1]",
    }, f"Cannot convert from source format {source}!"
    assert target in {
        "pil",
        "[0, 255]",
        "[0, 1]",
        "[-1, 1]",
        "imagenet-norm",
        "y-channel",
    }, f"Cannot convert to target format {target}!"

    if isinstance(img, Image.Image):
        assert source == "pil", "Source is not PIL, but input is"
        img = FT.to_tensor(img)

    match source:
        case "[0, 255]":
            img = img / 255.0
        case "[0, 1]":
            img = img
        case "[-1, 1]":
            img = (img + 1.0) / 2.0

    # Convert from [0, 1] to target
    match target:
        case "pil":
            img = FT.to_pil_image(img)
        case "[0, 255]":
            img = img * 255.0
        case "[0, 1]":
            img = img
        case "[-1, 1]":
            img = img * 2.0 - 1.0
        case "imagenet-norm":
            if img.ndimension() == 3:
                img = (img - imagenet_mean) / imagenet_std
            elif img.ndimension() == 4:
                img = (img - imagenet_mean_cuda) / imagenet_std_cuda
        case "y-channel":
            img = (
                torch.matmul(
                    255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights
                )
                / 255.0
                + 16.0
            )

    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {"train", "test"}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == "train":
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize(
            (
                int(hr_img.width / self.scaling_factor),
                int(hr_img.height / self.scaling_factor),
            ),
            Image.Resampling.BICUBIC,
        )

        # Sanity check
        assert (
            hr_img.width == lr_img.width * self.scaling_factor
            and hr_img.height == lr_img.height * self.scaling_factor
        )

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source="pil", target=self.lr_img_type)
        hr_img = convert_image(hr_img, source="pil", target=self.hr_img_type)

        return lr_img, hr_img
