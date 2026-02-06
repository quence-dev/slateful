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
    """Enhance image for better OCR"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(contrast, h=10)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Threshold to make text stand out
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

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
  
  if save_debug:
    print("\n--- All detected text: ---")
    for (bbox, text, conf) in results:
      print(f"  '{text}' (confidence: {conf:.2f})")
  
  # Try to parse scene/shot/take
  scene_pattern = r'(?:Scene|SC|S|scene|SCENE)[\s:\.#]*(\d+)\s*([A-Z]+(?:-\d+)?)?'
  take_pattern = r'(?:Take|TK|T|take|TAKE)[\s:\.#]*(\d+)'

  # Also try just finding standalone patterns (in case "Scene"/"Take" are vertical/separate)
  standalone_scene = r'\b(\d+)([A-Z]+(?:-\d+)?)\b'  # Matches "1A", "95B"
  standalone_take = r'\b[Tt](?:ake)?[\s:]*(\d+)\b'

  parsed_data = {}
  
  scene_match = re.search(scene_pattern, full_text, re.IGNORECASE)
  take_match = re.search(take_pattern, full_text, re.IGNORECASE)

  if scene_match:
    parsed_data['scene'] = scene_match.group(1)
    parsed_data['shot'] = scene_match.group(2) if scene_match.group(2) else ''
    print(f"Parsed: Scene {parsed_data['scene']}{parsed_data['shot']}")
  else:
    # Try standalone pattern (number + letter together)
    standalone_match = re.search(standalone_scene, full_text)
    if standalone_match:
      parsed_data['scene'] = standalone_match.group(1)
      parsed_data['shot'] = standalone_match.group(2)
      print(f"Parsed (standalone): Scene {parsed_data['scene']}{parsed_data['shot']}")
    else:
      print("Could not find scene/shot number")
  
  if take_match:
    parsed_data['take'] = take_match.group(1)
    print(f"Parsed: Take {parsed_data['take']}")
  else:
    standalone_match_take = re.search(standalone_take, full_text)
    if standalone_match_take:
      parsed_data['take'] = standalone_match_take.group(1)
      print(f"Parsed (standalone): Take {parsed_data['take']}")
    else:
      print("Could not find take number")
  
  return parsed_data

def load_labels():
  """Load ground truth labels"""
  labels_path = Path("test_data/slate_images/labels.json")
  
  if not labels_path.exists():
    print(f"Labels file not found: {labels_path}")
    print("Please create labels.json with ground truth data")
    exit(1)
  
  with open(labels_path, 'r') as f:
    return json.load(f)
    
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
    slate_folder = Path("test_data/slate_images")
    
    if not slate_folder.exists():
        print(f"Folder not found: {slate_folder}")
        print("Please create test_data/slate_images/ and add some slate images!")
        exit(1)

    labels = load_labels()
    print(f"Testing {len(labels)} labeled images\n")
    print("="*70)
    
    stats = {
      'total': 0,
      'scene_correct': 0,
      'shot_correct': 0,
      'take_correct': 0,
      'full_match': 0
    }

    for filename, expected in labels.items():
      img_file = slate_folder / filename

      if not img_file.exists():
        print(f"Image not found: {filename}")
        continue

      print(f"\n{filename}")
      # print(f"Expected: Scene {expected.get('scene', '?')}{expected.get('shot', '')}, Take {expected.get('take', '?')}")

      predicted = extract_slate_info(img_file)
      # print(f"Detected: Scene {predicted.get('scene', '?')}{predicted.get('shot', '')}, Take {predicted.get('take', '?')}")

      comparison = compare_results(predicted, expected)

      # Show results
      scene_status = "✅" if comparison['scene_correct'] else "❌"
      shot_status = "✅" if comparison['shot_correct'] else "❌"
      take_status = "✅" if comparison['take_correct'] else "❌"
      
      print(f"  Scene: {scene_status}  Shot: {shot_status}  Take: {take_status}")
      if comparison['full_match']:
        print(" PERFECT MATCH")

      # Update stats
      stats['total'] += 1
      if comparison['scene_correct']:
        stats['scene_correct'] += 1
      if comparison['shot_correct']:
        stats['shot_correct'] += 1
      if comparison['take_correct']:
        stats['take_correct'] += 1
      if comparison['full_match']:
        stats['full_match'] += 1
        
      print("-" * 70)
    
    # Final stats
    print(f"\nFINAL RESULTS")
    print(f"Scene accuracy: {stats['scene_correct']}/{stats['total']} ({stats['scene_correct']/stats['total']*100:.0f}%)")
    print(f"Shot accuracy: {stats['shot_correct']}/{stats['total']} ({stats['shot_correct']/stats['total']*100:.0f}%)")
    print(f"Take accuracy: {stats['take_correct']}/{stats['total']} ({stats['take_correct']/stats['total']*100:.0f}%)")
    print(f"Perfect matches: {stats['full_match']}/{stats['total']} ({stats['full_match']/stats['total']*100:.0f}%)")

    # image_files = list(slate_folder.glob("*.jpg")) + list(slate_folder.glob("*.png"))
    # # Preprocessed only:
    # # image_files = list(slate_folder.glob("preprocessed_*.jpg")) + list(slate_folder.glob("preprocessed_*.png"))
    
    # if not image_files:
    #     print("No images found in test_data/slate_images/")
    #     print("Please add some .jpg or .png slate images to test!")
    #     exit(1)
    
    # print(f"Found {len(image_files)} images to test\n")
    # print("="*50)

    # success_scene = 0
    # success_shot = 0
    # success_take = 0

    # for img_file in image_files:
    #     result = extract_slate_info(img_file)
    #     if result:
    #         if 'scene' in result:
    #             success_scene += 1
    #         if 'shot' in result:
    #             success_shot += 1
    #         if 'take' in result:
    #             success_take += 1
    #     print("-" * 50)
    
    # print(f"\nRESULTS")
    # print(f"Scene detected: {success_scene}/{len(image_files)} ({success_scene/len(image_files)*100:.0f}%)")
    # print(f"Shot detected: {success_shot}/{len(image_files)} ({success_shot/len(image_files)*100:.0f}%)")
    # print(f"Take detected: {success_take}/{len(image_files)} ({success_take/len(image_files)*100:.0f}%)")