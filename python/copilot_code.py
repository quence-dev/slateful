"""
Slateful Pipeline - Complete slate detection and OCR system
Detects slate fields using YOLO OBB, crops, preprocesses, and extracts scene/take data
"""

import cv2
import numpy as np
import re
from pathlib import Path
from ultralytics import YOLO
import easyocr
import test_ocr

# Initialize models (loads once on import)
print("Loading YOLO model...")
detector = YOLO('./models/yolo11n-obb_model.pt')
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])
print("Models loaded!")


# ============================================================================
# 1. DETECTION
# ============================================================================

def detect_slate_fields(image, confidence=0.5):
    """
    Run YOLO OBB detection on image.
    
    Args:
        image: numpy array (BGR image from cv2.imread)
        confidence: detection confidence threshold (0.0-1.0)
    
    Returns:
        dict: {
            'slate': {'box': obb_box, 'confidence': float} or None,
            'scene': {'box': obb_box, 'confidence': float} or None,
            'take': {'box': obb_box, 'confidence': float} or None
        }
    """
    results = detector(image, conf=confidence, verbose=False)
    
    detections = {'slate': None, 'scene': None, 'take': None}
    
    # Group detections by class
    temp_detections = {'slate': [], 'scene': [], 'take': []}
    
    for box in results[0].obb:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])
        
        if class_name in temp_detections:
            temp_detections[class_name].append({
                'box': box,
                'confidence': conf
            })
    
    # RULE: Take only highest confidence detection per class
    for class_name in ['slate', 'scene', 'take']:
        if temp_detections[class_name]:
            detections[class_name] = max(temp_detections[class_name], 
                                        key=lambda x: x['confidence'])
    
    return detections


# ============================================================================
# 2. CROPPING
# ============================================================================

# def crop_obb(image, obb_box):
    """
    Crop rotated bounding box from image.
    
    Args:
        image: numpy array (BGR image)
        obb_box: OBB box object from YOLO detection
    
    Returns:
        numpy array: cropped and rotated image region
    """
    # Get 4 corner points
    # points = obb_box.xyxyxyxy[0].cpu().numpy().astype(np.float32)
    print('obb_box: ', obb_box)
    points = obb_box.xyxyxyxy.cpu().numpy().reshape(-1, 2).astype(np.float32)
    print('points: ', points)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    print('rect:', rect)
    print('box: ', box)
    
    # Get width and height
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    # Swap if needed (ensure width > height for horizontal text)
    # if width < height:
    #     width, height = height, width
    
    # Source points (rotated rectangle corners)
    src_pts = box.astype("float32")
    
    # Destination points (axis-aligned rectangle)
    dst_pts = np.array([
        [0, height-1],
        [0, 0],
        [width-1, 0],
        [width-1, height-1]
    ], dtype="float32")
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def crop_obb(image, obb_box):
    """
    Crop the oriented bounding box directly using YOLO-provided geometry.
    """
    # Center, width, height, rotation (degrees) — already provided by YOLO
    center_x, center_y, box_w, box_h, angle_deg = (
        obb_box.xywhr[0].cpu().numpy().astype(np.float32)
    )

    # Axis-aligned bounds (for clamping after rotation)
    xyxy = obb_box.xyxy.cpu().numpy().squeeze().astype(int)
    xmin, ymin, xmax, ymax = xyxy

    # Rotate full image about the box center so the box becomes axis-aligned
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop the axis-aligned rectangle defined by the box width/height
    half_w = int(round(box_w / 2))
    half_h = int(round(box_h / 2))
    cx = int(round(center_x))
    cy = int(round(center_y))

    x_start = max(cx - half_w, 0, xmin)
    x_end = min(cx + half_w, rotated.shape[1], xmax)
    y_start = max(cy - half_h, 0, ymin)
    y_end = min(cy + half_h, rotated.shape[0], ymax)

    if x_start >= x_end or y_start >= y_end:
        return None  # degenerate box

    return rotated[y_start:y_end, x_start:x_end]

# ============================================================================
# 3. PREPROCESSING
# ============================================================================

def preprocess_image(image, enabled=True):
    """
    Apply preprocessing to improve OCR accuracy.
    
    Args:
        image: numpy array (can be BGR or grayscale)
        enabled: if False, returns original image
    
    Returns:
        numpy array: processed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not enabled:
        return gray
    
    # Mild contrast using CLAHE (lower clipLimit)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    
    # Subtle sharpening using a small kernel (no blurring step)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(contrast, -1, kernel)
    
    # Slight gamma correction to lift midtones (helps dark text)
    gamma = 1.03
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype('uint8')
    corrected = cv2.LUT(sharpened, look_up_table)
    
    # Normalize to full 0-255 range to help OCR contrast
    normalized = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized


# ============================================================================
# 3a. SAVE DEBUG IMAGE
# ============================================================================

def save_debug_image(image, filepath):
    """
    Save image to disk for debugging.
    
    Args:
        image: numpy array
        filepath: path to save (string or Path)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), image)
    print(f"Saved debug image: {filepath}")


# ============================================================================
# 4. OCR
# ============================================================================

def run_ocr(image, confidence=0.5):
    """
    Run EasyOCR on image.
    
    Args:
        image: numpy array (grayscale or BGR)
    
    Returns:
        str: concatenated text from all detected regions
    """
    results = reader.readtext(image, contrast_ths=0.5, adjust_contrast=0.5)  # detail=0 returns only text
    
    print("\n--- All Detected Text ---")
    for (bbox, text, conf) in results:
        print(f"  '{text}' (confidence: {conf:.2f})")

    text = ' '.join([text for (_, text, conf) in results if conf >= confidence])
    
    # Retry without confidence interval
    if not len(text):
        text = ' '.join([text for (_, text, _) in results])

    return text


# ============================================================================
# 5. PARSING
# ============================================================================

def parse_scene(text):
    """
    Extract scene number and optional shot letter from OCR text.
    Scene format: number + optional letter (e.g., "95A", "12", "3B")
    
    Args:
        text: OCR text string
    
    Returns:
        dict: {'scene': '95A'} or None if not found
    """
    # Pattern: number followed by optional letter(s)
    pattern = r'\b(\d+)([A-Z]+(?:-\d+)?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        scene_num = match.group(1)
        shot_letter = match.group(2).upper() if match.group(2) else ''
        
        # Combine into single scene value
        scene = scene_num + shot_letter
        
        # Validate: reasonable scene number
        if 1 <= int(scene_num) <= 9999:
            return {'scene': scene}
        else:
            print(f"  ⚠️  Scene number {scene_num} outside normal range (1-9999)")
            return {'scene': scene}  # Return anyway but warn
    
    return None


def parse_take(text):
    """
    Extract take number from OCR text.
    Take format: number only (e.g., "3", "12")
    
    Args:
        text: OCR text string
    
    Returns:
        dict: {'take': '3'} or None if not found
    """
    # Pattern: extract all numbers
    numbers = re.findall(r'\d+', text)
    
    if numbers:
        take_num = numbers[0]  # Take first number found
        
        # Validate: reasonable take number
        if 1 <= int(take_num) <= 999:
            return {'take': take_num}
        else:
            print(f"  ⚠️  Take number {take_num} outside normal range (1-999)")
            return {'take': take_num}  # Return anyway but warn
    
    return None


# ============================================================================
# 7. COMPARISON
# ============================================================================

def compare_results(detected, expected):
    """
    Compare detected values against expected (from filename).
    
    Args:
        detected: dict with 'scene' and 'take' keys
        expected: dict with 'scene' and 'take' keys
    
    Returns:
        dict: {
            'scene_correct': bool,
            'take_correct': bool,
            'full_match': bool
        }
    """
    scene_correct = detected.get('scene') == expected.get('scene')
    take_correct = detected.get('take') == expected.get('take')
    
    return {
        'scene_correct': scene_correct,
        'take_correct': take_correct,
        'full_match': scene_correct and take_correct
    }


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def process_slate_image(image_path, img_confidence=0.5, txt_confidence=0.5, preprocess_enabled=True, save_debug=False):
    """
    Complete pipeline: detect → crop → preprocess → OCR → parse → validate.
    
    Args:
        image_path: path to image file
        confidence: YOLO detection confidence threshold (0.0-1.0)
        preprocess_enabled: apply preprocessing before OCR
        save_debug: save intermediate cropped/processed images
    
    Returns:
        dict: {
            'filename': str,
            'detected': {'scene': str, 'take': str},
            'expected': {'scene': str, 'take': str},
            'comparison': {...},
            'raw_ocr': {'scene': str, 'take': str},
            'detection_confidences': {'scene': float, 'take': float}
        }
    """
    image_path = Path(image_path)
    print(f"\nProcessing: {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ❌ Could not read image")
        return None
    
    # Step 1: Detect
    detections = detect_slate_fields(img, confidence=img_confidence)
    
    if not detections['scene'] or not detections['take']:
        print(f"  ❌ Detection failed:")
        print(f"     Scene detected: {detections['scene'] is not None}")
        print(f"     Take detected: {detections['take'] is not None}")
        return None
    
    print(f"Detections: scene={detections['scene']['confidence']:.2f}, take={detections['take']['confidence']:.2f}")
    
    # Step 2-5: Process scene
    scene_crop = crop_obb(img, detections['scene']['box'])
    
    # Add padding
    scene_crop = cv2.copyMakeBorder(scene_crop, 5, 5, 5, 5, 
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    if save_debug:
        save_debug_image(scene_crop, f"test_data/debug/{image_path.stem}_scene_crop.png")
    
    if preprocess_enabled:
        scene_crop = preprocess_image(scene_crop, enabled=preprocess_enabled)
        if save_debug:
            save_debug_image(scene_crop, f"test_data/debug/{image_path.stem}_scene_processed.png")
    
    scene_text = run_ocr(scene_crop, txt_confidence)
    print(f"Scene OCR: '{scene_text}'")
    
    scene_parsed = parse_scene(scene_text)
    
    # Step 2-5: Process take
    take_crop = crop_obb(img, detections['take']['box'])
    
    # Add padding
    take_crop = cv2.copyMakeBorder(take_crop, 5, 5, 5, 5,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    if save_debug:
        save_debug_image(take_crop, f"test_data/debug/{image_path.stem}_take_crop.png")
    
    if preprocess_enabled:
        take_crop = preprocess_image(take_crop, enabled=preprocess_enabled)
        if save_debug:
            save_debug_image(take_crop, f"test_data/debug/{image_path.stem}_take_processed.png")
    
    take_text = run_ocr(take_crop, txt_confidence)
    print(f"Take OCR: '{take_text}'")
    
    take_parsed = parse_take(take_text)
    
    # Step 6: Parse filename
    expected = test_ocr.parse_data_from_filename(image_path)
    
    if not expected:
        print(f"Could not parse filename")
        expected = {'scene': '?', 'take': '?'}
    
    # Combine detected results
    detected = {
        'scene': scene_parsed['scene'] if scene_parsed else None,
        'take': take_parsed['take'] if take_parsed else None
    }
    
    # Step 7: Compare
    comparison = compare_results(detected, expected)
    
    # Print results
    scene_icon = "✅" if comparison['scene_correct'] else "❌"
    take_icon = "✅" if comparison['take_correct'] else "❌"
    
    print(f"  {scene_icon} Scene: detected={detected['scene']}, expected={expected['scene']}")
    print(f"  {take_icon} Take:  detected={detected['take']}, expected={expected['take']}")
    
    # Return complete results
    return {
        'filename': image_path.name,
        'detected': detected,
        'expected': expected,
        'comparison': comparison,
        'raw_ocr': {
            'scene': scene_text,
            'take': take_text
        },
        'detection_confidences': {
            'scene': detections['scene']['confidence'],
            'take': detections['take']['confidence']
        }
    }


# ============================================================================
# BATCH TESTING
# ============================================================================

def test_directory(directory, img_confidence=0.5, txt_confidence=0.5, preprocess_enabled=True, save_debug=False):
    """
    Test all images in a directory and report accuracy.
    
    Args:
        directory: path to folder containing test images
        img_confidence: YOLO confidence threshold
        txt_confidence: OCR confidence threshold
        preprocess_enabled: apply preprocessing
        save_debug: save debug images
    
    Returns:
        dict: summary statistics
    """
    directory = Path(directory)
    image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {directory}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing {len(image_files)} images from {directory}")
    print(f"YOLO Confidence: {img_confidence}, OCR Confidence: {txt_confidence}, Preprocessing: {preprocess_enabled}")
    print(f"{'='*60}")
    
    results = []
    scene_correct = 0
    take_correct = 0
    full_match = 0
    failed_detections = 0
    
    for img_file in image_files:
        result = process_slate_image(img_file, img_confidence, txt_confidence, preprocess_enabled, save_debug)
        
        if result is None:
            failed_detections += 1
            continue
        
        results.append(result)
        
        if result['comparison']['scene_correct']:
            scene_correct += 1
        if result['comparison']['take_correct']:
            take_correct += 1
        if result['comparison']['full_match']:
            full_match += 1
    
    # Summary
    total = len(image_files)
    successful = len(results)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:        {total}")
    print(f"Successful detections: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed detections:   {failed_detections}")
    print(f"\nAccuracy (of successful detections):")
    if successful > 0:
        print(f"  Scene accuracy:    {scene_correct}/{successful} ({scene_correct/successful*100:.1f}%)")
        print(f"  Take accuracy:     {take_correct}/{successful} ({take_correct/successful*100:.1f}%)")
        print(f"  Full match:        {full_match}/{successful} ({full_match/successful*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        'total': total,
        'successful': successful,
        'failed_detections': failed_detections,
        'scene_correct': scene_correct,
        'take_correct': take_correct,
        'full_match': full_match,
        'results': results
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test on a single image
    # result = process_slate_image(
    #     "test_data/slate_images/95A_take3.jpg",
    #     confidence=0.5,
    #     preprocess_enabled=True,
    #     save_debug=True
    # )
    
    # Test on entire directory
    summary = test_directory(
        "test_data/slate_images/uncropped",
        img_confidence=0.6,
        txt_confidence=0.5,
        preprocess_enabled=True,
        save_debug=True  # Set to True to save crops
    )