# Part 3. Extract features from images

The cvObjParser.py file takes two arguments, one for image output, and the other for the folder containing all the images for a listing. 

The script uses a pre-compiled computer vision model from a python library. The script will find  furniture, appliances, and plants, as we predict a listing with furniture will get more interest than a listing without. The script will also extract contrast values, and count the number of images. We predict that high contrast values mean listing will have post processing, and the more images a listing has will create more interest than a listing with no images, or a few images.

## Output images 

![object detection image 1](./obj_detection_imgs/object-detection_1.jpg)
![object detection image 2](./obj_detection_imgs/object-detection_2.png)
![object detection image 3](./obj_detection_imgs/object-detection_3.png)
![object detection image 4](./obj_detection_imgs/object-detection_4.jpg)
![object detection image 5](./obj_detection_imgs/object-detection_5.jpg)

## Feature extraction output

### Output for one listing
![output_image 1](./obj_detection_imgs/output_1.png)

### Output for all listings (took ~10 mins on sample images)
![output_image 2](./obj_detection_imgs/output_2.png)

## Code

```
# cvObjParser.py

import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import numpy as np
import sys
import os

# usage: python3 cvObjParser.py [0:no output, 1:output] [image folder path]

# check if want to see image output with first argument (iterate through images with 'q')
check_output = sys.argv[1]

# maps frequency of features
def frequency(features):
    freq = {}
    for feature in features:
        freq[feature] = features.count(feature)
    #for key, value in freq.items():
        #print("% s: % s"%(key, value))
    return freq

# take image folder as argument & extract image paths
folder_path = sys.argv[2]
images = []
for r, d, f in os.walk(folder_path):
    for file in f:
        if '.jpg' in file:
            images.append(os.path.join(r, file))

# extract features from collection of images
img_contrasts = []
features = []
for image in images:
    
    in_img = cv2.imread(image)

    # show original image
    '''
    cv2.imshow('Image', in_img)
    cv2.waitKey()
    '''
 
    # check contrast
    c = cv2.cvtColor(in_img, cv2.COLOR_BGR2YUV)[:,:,0]
    c_min = float(np.min(c))
    c_max = float(np.max(c))
    contrast = (c_max-c_min)/(c_max+c_min)
    img_contrasts.append(round(contrast, 2))
    
    # perform analysis
    bbox, label, conf = cv.detect_common_objects(in_img)
    out_img = draw_bbox(in_img, bbox, label, conf)
    
    # show output image
    '''
    '''
    if check_output == '1':
        cv2.imshow('labeled', out_img)
        cv2.waitKey()
    
    # show features extracted
    for l in label:
        features.append(l)

# filter features into important ones
f = frequency(features)
print('\nImages have the following features & frequencies\n================================================\n')
print(f)

# check if listing furnished
furn_count = 0
filt_feat = ['bench', 'chair', 'couch', 'bed', 'dining table', 'tv', 'clock', 'vase']
for item in filt_feat:
    if item in f:
        furn_count = furn_count+f[item]
print('\nthere are', furn_count, 'pieces of furniture')

# check for plants
plant_count = 0
filt_feat = 'potted plant'
if filt_feat in f:
    plant_count = plant_count+f[filt_feat]
print('there are', plant_count, 'plants')

# check contrast (avg < 0.9 not as nice looking)
avg = 0
for cont in img_contrasts:
    avg = avg+cont
avg = round(avg/len(img_contrasts), 2)
print('image contrast values are', img_contrasts, 'with an average of', avg)

# check if bathroom or kitchen is shown
kitchen = False
filt_feat = ['microwave', 'oven', 'toaster', 'refrigerator']
for item in filt_feat:
    if item in f:
        kitchen = True
if kitchen == True:
    print('Kitchen shown')
else: 
    print('Kitchen not shown')
bathroom = False
if 'toilet' in f:
    bathroom = True
    print('Bathroom shown')
else: 
    print('Bathroom not shown')


# check other features
print('there are', len(images), 'pictures')
```

