# OCR Testing Results

## Test Date: 02-06-2026

Tested 10 slate images with EasyOCR.

### Success Rate
- Scene number: 1/10 parsed, incorrectly read (0%)
- Shot letter: 0/10 (0%)
- Take number: 1/10 (10%)

### Issues Found
- Missing most handwritten lettering
- Struggles with blurry or dark footage

### Conclusion
EasyOCR struggles with handwriting. Going to run a second test with updated regex, hopefully better results
- Better contrast preprocessing
- Better regex. We aren't necessarily looking for the word "scene" or "take"