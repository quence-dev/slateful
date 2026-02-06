# test_ocr.py - Test OCR on slate images

import easyocr
import cv2
from pathlib import Path
import re

# Initialize (takes ~10 seconds first time)
reader = easyocr.Reader(['en'], gpu=False)

def extract_slate_info(image_path):
    """Test OCR on a single slate image"""
    
    # Read image
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Could not read {image_path}")
        return None
    
    # Run OCR
    results = reader.readtext(img)
    
    # Combine all text
    full_text = ' '.join([text for (bbox, text, conf) in results])
    
    print(f"\nImage: {image_path.name}")
    print(f"Raw OCR: {full_text}")
    
    # Try to parse scene/shot/take
    scene_pattern = r'(?:Scene|SC|S)\s*[:#]?\s*(\d+)([A-Z]?)'
    take_pattern = r'(?:Take|TK|T)\s*[:#]?\s*(\d+)'
    
    scene_match = re.search(scene_pattern, full_text, re.IGNORECASE)
    take_match = re.search(take_pattern, full_text, re.IGNORECASE)
    
    parsed_data = {}
    
    if scene_match:
        scene_num = scene_match.group(1)
        shot = scene_match.group(2) if scene_match.group(2) else ''
        parsed_data['scene'] = scene_num
        parsed_data['shot'] = shot
        print(f"Parsed: Scene {scene_num}, Shot {shot}")
    else:
        print("Could not find scene number")
    
    if take_match:
        take_num = take_match.group(1)
        parsed_data['take'] = take_num
        print(f"Parsed: Take {take_num}")
    else:
        print("Could not find take number")
    
    return parsed_data

if __name__ == "__main__":
    # Test on slate images
    slate_folder = Path("test_data/slate_images")
    
    if not slate_folder.exists():
        print(f"Folder not found: {slate_folder}")
        print("Please create test_data/slate_images/ and add some slate images!")
        exit(1)
    
    image_files = list(slate_folder.glob("*.jpg")) + list(slate_folder.glob("*.png"))
    
    if not image_files:
        print("No images found in test_data/slate_images/")
        print("Please add some .jpg or .png slate images to test!")
        exit(1)
    
    print(f"Found {len(image_files)} images to test\n")
    print("="*50)
    
    for img_file in image_files:
        extract_slate_info(img_file)
        print("-" * 50)