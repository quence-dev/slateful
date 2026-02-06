# OCR Testing Results

## Test Date: 02-06-2026 - 002

Tested 10 slate images with EasyOCR. Improved parsing, preprocessing will be next.

### Results
- Scene number: 6/10 parsed, 4/10 correct (40%)
- Shot letter: 5/10 parsed, 4/10 correct (40%)
- Take number: 1/10 (10%)

### Issues
- Still struggles parsing the handwriting
- Struggles with blurry or dark footage

### Conclusions
- Better regex made it easier to capture the values it was able to successfully parse

## Test Date: 02-06-2026 - 001

Tested 10 slate images with EasyOCR.

### Results
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