# OCR Testing Results


## Test Date: 02-06-2026 - 004

Tested 10 slate images with EasyOCR. Used the preprocessor this time, as well as adjusted the shot letter to be parsed as lower- or upper-case.

### Results
- Scene number: 8/10 correct (80%)
- Shot letter: 8/10 correct (80%)
- Take number: 2/10 correct (20%)
- Perfect matches: 2/10 (20%)

### Issues
- Inconsistent parsing with takes

### Conclusions
- Will now move to training either image recognition model (since OCR is a bit better when it knows where to look) or refine handwriting parsing with custom OCR model.


## Test Date: 02-06-2026 - 003

Tested 10 slate images with EasyOCR. This time, I cropped the images directly to the areas where the scene and takes are, to see if the issue was with OCR or something else. I also improved the parsing by making it only look for a take after the scene/shot were removed from the text. Preprocessing was a bust and made most of the images completely illegible.

### Results
- Scene number: 4/10 correct (40%)
- Shot letter: 3/10 correct (30%)
- Take number: 4/10 correct (40%)

### Issues
- Inconsistent parsing.
- Still struggles with handwriting.

### Conclusions
- Will now move to training a model.


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