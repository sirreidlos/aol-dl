import os
import json
import warnings
from PIL import Image, UnidentifiedImageError

# Try enabling decompression bomb warnings as errors, if available
DecompressionBombWarning = getattr(Image, "DecompressionBombWarning", None)
if DecompressionBombWarning:
    warnings.simplefilter("error", DecompressionBombWarning)


def is_valid_image(img_path, min_size):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(img_path) as img:
                img.verify()

        with Image.open(img_path) as img:
            if img.width < min_size or img.height < min_size:
                return False, "Too small"

        return True, None

    except (UnidentifiedImageError, OSError, Warning) as e:
        return False, str(e)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    print("\nCreating data lists... this may take some time.\n")
    filtered_out = []
    train_images = []
    test_images = []

    for d in train_folders:
        for filename in os.listdir(d):
            img_path = os.path.join(d, filename)
            valid, reason = is_valid_image(img_path, min_size)
            if valid:
                train_images.append(img_path)
            else:
                filtered_out.append((img_path, reason))

    print(f"There are {len(train_images)} images in the training data.\n")
    with open(os.path.join(output_folder, "train_images.json"), "w") as f:
        json.dump(train_images, f, indent=2)

    for d in test_folders:
        for filename in os.listdir(d):
            img_path = os.path.join(d, filename)
            valid, reason = is_valid_image(img_path, min_size)
            if valid:
                test_images.append(img_path)
            else:
                filtered_out.append((img_path, reason))

    print(f"There are {len(test_images)} images in the testing data.\n")
    with open(os.path.join(output_folder, "test_images.json"), "w") as f:
        json.dump(test_images, f, indent=2)

    # Summary logging
    if filtered_out:
        print("\nFiltered out (first 30 shown):")
        for path, reason in filtered_out[:30]:
            print(f"- {path}: {reason}")
        print(f"\nTotal filtered out: {len(filtered_out)}")
    else:
        print("\n✅ No filtered-out images!")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    create_data_lists(
        train_folders=[],
        test_folders=["./data/danbooru_val"],
        min_size=100,
        output_folder="./",
    )
