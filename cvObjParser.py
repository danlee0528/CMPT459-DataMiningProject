import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import numpy as np
import sys
import os
from collections import defaultdict
import json
import pandas as pd

# TODO
# () output json
# () make json row
# () add data
# ()
# ()

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
    '''
    for dir in d:
        print('directory is', dir)
    '''
   
    for file in f:
        if '.jpg' in file:
            images.append(os.path.join(r, file))
            #list_id[dir]= os.path.join(r, file)

# create dictionary for listings with images
listing_id = defaultdict(list)
for img in images:
    sub_str = img.split('/')
    #print(sub_str)
    #listing_id[sub_str[1]] = []
    listing_id[sub_str[1]].append(sub_str[2])
    
print('listing data is', listing_id)

# extract features from collection of images
json_data = pd.DataFrame({'listing_id':[], 'num_images':[], 'num_plants':[], 'kit_img': [], 'bath_img':[], 'num_furniture':[], 'contrast': []})
for (listing, image_list) in listing_id.items():
    img_contrasts = []
    features = []
    for image in image_list:
        #print('listing is', listing)
        #print('image is', image)
        im_path = folder_path + listing + '/' + image
        #print('image path:', im_path)
        
        in_img = cv2.imread(im_path)
    
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

    # make dictionary to convert to json 
    row = pd.DataFrame({'listing_id':[listing], 'num_images':[len(image_list)], 'num_plants':[plant_count], 'kit_img': [kitchen], 'bath_img':[bathroom], 'num_furniture':[furn_count], 'contrast': [avg]})
    #json_data['data'].append(row)
    json_data = json_data.append(row, ignore_index=True, sort=False)

# final output and writing to json
#json_obj = json.dumps(json_data)
print(json_data)

json_data.to_json('image_data.json', orient='records')
