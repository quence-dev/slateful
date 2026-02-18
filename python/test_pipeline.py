from pathlib import Path

def parse_data_from_filename(filename, include_extra_data = False):
    # Accept either a string or a pathlib.Path; operate on the name
    if isinstance(filename, Path):
        name = filename.stem
    else:
        name = Path(str(filename)).stem

    split_name = name.split()
    length = len(split_name)

    data = {}

    if length == 4:
        # used for "98 E - 01" syntax
        data["scene"] = "".join(split_name[0:2])
        data["take"] = split_name[-1].lstrip("0")
    elif length < 4:
        # used for "17A T2" syntax
        data["scene"] = split_name[0]
        data["take"] = split_name[1].replace("T", "")
        # I noticed occasionally I have a file like "18H T1 MOS"
        if length == 3 and include_extra_data:
            data["other"] = split_name[2]

    return data

# quick test to see if it works
if __name__ == "__main__":
    slate_folder = Path("training_data/raw_slates")
    include_extra_data = False

    if not slate_folder.exists():
        print(f"Folder not found: {slate_folder}")
        exit(1)

    image_files = list(slate_folder.glob("*.jpg")) + list(slate_folder.glob("*.png"))

    if not image_files:
        print(f"No images found in {slate_folder}")
        exit(1)

    for img_file in image_files:
        data = parse_data_from_filename(img_file, include_extra_data)
        print(f"Filename: {img_file}")
        print(f"Scene: {data.get('scene')}")
        print(f"Take: {data.get('take')}")
        if "other" in data and include_extra_data:
            print(f"Other: {data.get('other')}")
            print("="*70)
        print("\n")
        