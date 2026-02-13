from pathlib import Path

def parse_data_from_filename(filename):
    # Accept either a string or a pathlib.Path; operate on the name
    if isinstance(filename, Path):
        name = filename.stem
    else:
        name = Path(str(filename)).stem

    split_name = name.split()

    data = {}

    if len(split_name) == 4:
        # used for "98 E - 01" syntax
        data["scene"] = "".join(split_name[0:2])
        data["take"] = split_name[-1].lstrip("0")
    elif len(split_name) == 2:
        # used for "17A T2" syntax
        data["scene"] = split_name[0]
        data["take"] = split_name[1].replace("T", "")

    return data


# quick test to see if it works
if __name__ == "__main__":
    slate_folder = Path("training_data/raw_slates")

    if not slate_folder.exists():
        print(f"Folder not found: {slate_folder}")
        exit(1)

    image_files = list(slate_folder.glob("*.jpg")) + list(slate_folder.glob("*.png"))

    if not image_files:
        print(f"No images found in {slate_folder}")
        exit(1)

    for img_file in image_files:
        data = parse_data_from_filename(img_file.stem)
        print(f"Filename: {img_file}")
        print(f"Scene: {data.get('scene')}")
        print(f"Take: {data.get('take')} \n")