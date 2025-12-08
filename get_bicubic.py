import sys
from pathlib import Path
from PIL import Image, ImageOps


def bicubic_downsample(input_path: Path, factor: int = 4) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.is_dir():
        raise IsADirectoryError(f"Input is a directory, expected a file: {input_path}")

    with Image.open(input_path) as im:
        if im.mode in ("P",):
            im = im.convert("RGBA")
        im = ImageOps.exif_transpose(im)

        w, h = im.size
        new_w = max(1, w // factor)
        new_h = max(1, h // factor)

        im_resized = im.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        out_name = f"{input_path.stem}_bicubic{input_path.suffix}"
        out_path = input_path.with_name(out_name)

        save_kwargs = {}
        im_resized.save(out_path, **save_kwargs)

    return out_path


def main(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(
        description="Save a 4x bicubic-downsampled copy of an image."
    )
    p.add_argument("image", type=Path, help="Path to the input image")
    p.add_argument(
        "-f", "--factor", type=int, default=4, help="Downsample factor (default: 4)"
    )
    args = p.parse_args(argv)

    try:
        out = bicubic_downsample(args.image, factor=args.factor)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
