import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import sys

# take image as argument (must be in current folder)
img_path = sys.argv[1] 
in_img = cv2.imread(img_path)

# show original image
cv2.imshow('orignal', in_img)
cv2.waitKey()

# perform analysis
bbox, label, conf = cv.detect_common_objects(in_img)
out_img = draw_bbox(in_img, bbox, label, conf)

# show output image
cv2.imshow('labeled', out_img)
cv2.waitKey()

# show features extracted
print('features:', label)
