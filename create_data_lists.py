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
        train_folders=["./data/danbooru"],
        test_folders=[],
        min_size=100,
        output_folder="./",
    )

# import os
# import json
# from PIL import Image, UnidentifiedImageError
# import warnings


# def create_data_lists(train_folders, test_folders, min_size, output_folder):
#     print("\nCreating data lists... this may take some time.\n")
#     filtered_out = []

#     train_images = []
#     for d in train_folders:
#         for i in os.listdir(d):
#             img_path = os.path.join(d, i)
#             try:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("error")  # Treat EXIF warnings as errors
#                     img = Image.open(img_path)
#                     img.verify()  # Validate file integrity

#                 img = Image.open(img_path)  # Reopen after verify
#                 if img.width >= min_size and img.height >= min_size:
#                     train_images.append(img_path)
#                 else:
#                     filtered_out.append((img_path, "Too small"))
#             except (UnidentifiedImageError, OSError, Warning) as e:
#                 filtered_out.append((img_path, str(e)))

#     print("There are %d images in the training data.\n" % len(train_images))
#     with open(os.path.join(output_folder, "train_images.json"), "w") as j:
#         json.dump(train_images, j, indent=2)

#     # Test folders section (same protection added)
#     for d in test_folders:
#         test_images = []
#         test_name = os.path.basename(d)
#         for i in os.listdir(d):
#             img_path = os.path.join(d, i)
#             try:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("error")
#                     img = Image.open(img_path)
#                     img.verify()

#                 img = Image.open(img_path)
#                 if img.width >= min_size and img.height >= min_size:
#                     test_images.append(img_path)
#                 else:
#                     filtered_out.append((img_path, "Too small"))
#             except (UnidentifiedImageError, OSError, Warning) as e:
#                 filtered_out.append((img_path, str(e)))

#         print(f"There are {len(test_images)} images in the {test_name} test data.\n")
#         with open(
#             os.path.join(output_folder, f"{test_name}_test_images.json"), "w"
#         ) as j:
#             json.dump(test_images, j, indent=2)

#     print("JSON files saved to:", output_folder)

#     # Report skipped files
#     if filtered_out:
#         print("\nFiltered out images (with reasons):")
#         for img, reason in filtered_out[:30]:  # Avoid huge spam
#             print(f"- {img}: {reason}")
#         print(f"\nTotal filtered out: {len(filtered_out)}")
#     else:
#         print("No problematic images found ✅")


# if __name__ == "__main__":
#     create_data_lists(
#         train_folders=["./data/danbooru"],
#         test_folders=[],
#         min_size=100,
#         output_folder="./",
#     )
