# test_ocr.py - Test OCR on slate images

import easyocr
import cv2
from pathlib import Path
import re
import json
import numpy as np

# Initialize (takes ~10 seconds first time)
reader = easyocr.Reader(['en'])

def preprocess_image(img):
  """Lighter preprocessing for OCR: mild contrast and subtle sharpening,
  without any added blur or heavy denoising. Returns grayscale image
  optimized for OCR."""

  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Mild contrast using CLAHE (lower clipLimit)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  contrast = clahe.apply(gray)

  # Subtle sharpening using a small kernel (no blurring step)
  kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=np.float32)
  sharpened = cv2.filter2D(contrast, -1, kernel)

  # Slight gamma correction to lift midtones (helps dark text)
  gamma = 1.03
  look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype('uint8')
  corrected = cv2.LUT(sharpened, look_up_table)

  # Normalize to full 0-255 range to help OCR contrast
  normalized = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

  return normalized

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

def crop_rotated_box(image, obb_box):
    """Crop rotated bounding box from image"""
    
    # Get 4 corner points
    points = obb_box.xyxyxyxy[0].cpu().numpy().astype(np.float32)
    
    # Get bounding rectangle
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get width and height of the rotated rect
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    # Get rotation matrix
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

# TODO: deprecate
def extract_slate_info(image_path, save_debug=False):
  """Test OCR on a single slate image"""
  
  # Read image
  img = cv2.imread(str(image_path))
  
  if img is None:
    print(f"Could not read {image_path}")
    return None
  
  # Run OCR
  results = reader.readtext(img)
  full_text = ' '.join([text for (bbox, text, conf) in results])
  print(f"\nRaw OCR: {full_text}")

  # print("\n--- Preprocessed Image ---")
  # preprocessed = preprocess_image(img)
  # # SAVE PREPROCESSED IMAGE TO VIEW IT
  # output_path = f"test_data/slate_images/preprocessed_{image_path.name}"
  # cv2.imwrite(output_path, preprocessed)
  # #####
  # results_preprocessed = reader.readtext(preprocessed)
  # text_preprocessed = ' '.join([text for (bbox, text, conf) in results_preprocessed])
  # print(f"Raw OCR: {text_preprocessed}")

  # full_text = text_preprocessed
  
  if save_debug:
    print("\n--- All detected text: ---")
    for (bbox, text, conf) in results:
      print(f"  '{text}' (confidence: {conf:.2f})")

  scene_pattern = r'\b(\d+)([A-Z]+(?:-\d+)?)\b'
  take_pattern  = r'\b([1-9][0-9]?)\b(?!\s*[A-Za-z-])'

  parsed_data = {}
  
  scene_match = re.search(scene_pattern, full_text, re.IGNORECASE)
  remaining = full_text

  if scene_match:
    parsed_data['scene'] = scene_match.group(1)
    shot_val = scene_match.group(2) or ''
    parsed_data['shot'] = shot_val.upper()
    start, end = scene_match.span()
    remaining = full_text[:start] + ' ' + full_text[end:]
    print(f"Parsed: Scene {parsed_data['scene']}{parsed_data['shot']}")
  else:
    print("Could not find scene/shot number")
  
  print(f"\n Checking in: {remaining}")
  take_match = re.search(take_pattern, remaining, re.IGNORECASE)
  if take_match:
    parsed_data['take'] = take_match.group(1)
    print(f"Parsed: Take {parsed_data['take']}")
  else:
    print("Could not find take number")
  
  return parsed_data

# TODO: deprecate
def load_labels():
  """Load ground truth labels"""
  labels_path = Path("test_data/slate_images/labels.json")
  
  if not labels_path.exists():
    print(f"Labels file not found: {labels_path}")
    print("Please create labels.json with ground truth data")
    exit(1)
  
  with open(labels_path, 'r') as f:
    return json.load(f)
    
# TODO: rewrite
def compare_results(predicted, expected):
  """Compare predicted vs expected values"""
  results = {
    'scene_correct': predicted.get('scene') == expected.get('scene'),
    'shot_correct': predicted.get('shot') == expected.get('shot'),
    'take_correct': predicted.get('take') == expected.get('take'),
    'full_match': (
      predicted.get('scene') == expected.get('scene') and
      predicted.get('shot') == expected.get('shot') and
      predicted.get('take') == expected.get('take')
    )
  }
  return results

if __name__ == "__main__":
    # slate_folder = Path("test_data/slate_images")
    slate_folder = Path("training_data/raw_slates")
    
    if not slate_folder.exists():
        print(f"Folder not found: {slate_folder}")
        print("Please create test_data/slate_images/ and add some slate images!")
        exit(1)

    # labels = load_labels()
    # print(f"Testing {len(labels)} labeled images\n")
    # print("="*70)
    
    # stats = {
    #   'total': 0,
    #   'scene_correct': 0,
    #   'shot_correct': 0,
    #   'take_correct': 0,
    #   'full_match': 0
    # }

    # for filename, expected in labels.items():
    #   # preproc_name = f"preprocessed_{filename}"
    #   img_file = slate_folder / filename

    #   if not img_file.exists():
    #     print(f"Image not found: {filename}")
    #     continue

    #   print(f"\n{filename}")
    #   # print(f"Expected: Scene {expected.get('scene', '?')}{expected.get('shot', '')}, Take {expected.get('take', '?')}")

    #   predicted = extract_slate_info(img_file)
    #   # print(f"Detected: Scene {predicted.get('scene', '?')}{predicted.get('shot', '')}, Take {predicted.get('take', '?')}")

    #   comparison = compare_results(predicted, expected)

    #   # Show results
    #   scene_status = "✅" if comparison['scene_correct'] else "❌"
    #   shot_status = "✅" if comparison['shot_correct'] else "❌"
    #   take_status = "✅" if comparison['take_correct'] else "❌"
      
    #   print(f"  Scene: {scene_status}  Shot: {shot_status}  Take: {take_status}")
    #   if comparison['full_match']:
    #     print(" PERFECT MATCH")

    #   # Update stats
    #   stats['total'] += 1
    #   if comparison['scene_correct']:
    #     stats['scene_correct'] += 1
    #   if comparison['shot_correct']:
    #     stats['shot_correct'] += 1
    #   if comparison['take_correct']:
    #     stats['take_correct'] += 1
    #   if comparison['full_match']:
    #     stats['full_match'] += 1
        
    #   print("-" * 70)
    
    # # Final stats
    # print(f"\nFINAL RESULTS")
    # print(f"Scene accuracy: {stats['scene_correct']}/{stats['total']} ({stats['scene_correct']/stats['total']*100:.0f}%)")
    # print(f"Shot accuracy: {stats['shot_correct']}/{stats['total']} ({stats['shot_correct']/stats['total']*100:.0f}%)")
    # print(f"Take accuracy: {stats['take_correct']}/{stats['total']} ({stats['take_correct']/stats['total']*100:.0f}%)")
    # print(f"Perfect matches: {stats['full_match']}/{stats['total']} ({stats['full_match']/stats['total']*100:.0f}%)")

    image_files = list(slate_folder.glob("*.jpg")) + list(slate_folder.glob("*.png"))
    # # Preprocessed only:
    # image_files = list(slate_folder.glob("preprocessed_*.jpg")) + list(slate_folder.glob("preprocessed_*.png"))
    
    if not image_files:
        print("No images found in test_data/slate_images/")
        print("Please add some .jpg or .png slate images to test!")
        exit(1)
    
    print(f"Found {len(image_files)} images to test\n")
    print("="*50)

    success_scene = 0
    success_shot = 0
    success_take = 0

    for img_file in image_files:
        result = extract_slate_info(img_file)
        if result:
            if 'scene' in result:
                success_scene += 1
            if 'shot' in result:
                success_shot += 1
            if 'take' in result:
                success_take += 1
        print("-" * 50)
    
    print(f"\nRESULTS")
    print(f"Scene detected: {success_scene}/{len(image_files)} ({success_scene/len(image_files)*100:.0f}%)")
    print(f"Shot detected: {success_shot}/{len(image_files)} ({success_shot/len(image_files)*100:.0f}%)")
    print(f"Take detected: {success_take}/{len(image_files)} ({success_take/len(image_files)*100:.0f}%)")