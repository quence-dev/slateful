# Purpose: Parse OCR text into structured data
# Example:

# import re

# def parse_slate_text(ocr_text):
#     """Extract scene, shot, take from OCR text"""
    
#     # Find "Scene 95A"
#     scene_match = re.search(r'Scene\s*(\d+)([A-Z]?)', ocr_text)
    
#     # Find "Take 3"
#     take_match = re.search(r'Take\s*(\d+)', ocr_text)
    
#     return {
#         'scene': scene_match.group(1) if scene_match else None,
#         'shot': scene_match.group(2) if scene_match else '',
#         'take': take_match.group(1) if take_match else None
#     }