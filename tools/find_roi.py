from glob import glob
import cv2 
import numpy as np
import os 
from shapedetector import ShapeDetector
import argparse
import imutils
from xml_tools import *

mask_paths = glob('/Users/macos/git_project/thesis2/thesis/output/*_output.png')
input_paths = [path.split('output.png')[0]+'input.png' for path in mask_paths]



# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
def draw_image(input_image, mask_image):
    resized = mask_image#imutils.resize(mask_image, width=300)
    ratio = mask_image.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    pad = np.zeros_like(input_image).astype(np.uint8)
    cells = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < 100 and h > 20 and w > 20:    
            pad[y:y+h, x:x+w] = input_image[y:y+h, x:x+w]
            xmin, ymin = x,y
            xmax, ymax = x+w, y+h 
            cells.append(Bndbox(xmin=xmin,
                       xmax=xmax,
                       ymin=ymin,
                       ymax=ymax,
                       ))
    return mask_image, pad, cells

if __name__ == '__main__':
    assert len(mask_paths) == len(input_paths) and len(input_paths) > 0
    print('Len:', len(mask_paths))
    for i, (in_path, ma_path) in enumerate(zip(input_paths, mask_paths)):
        # ma_path = 'temp/322/0/output.png'
        # in_path = 'temp/322/0/input.png'
        mask_image =  cv2.imread(ma_path)
        input_image = cv2.imread(in_path)
        print(mask_image.shape==input_image.shape)
        _, pad, cells = draw_image(input_image, mask_image)
        img_path = 'output/text_line/{i:03d}.png'.format(i=i)
        cv2.imwrite(img_path, input_image)
        xml = cells_to_xml(cells)
        with open('output/text_line/{i:03d}.xml'.format(i=i), mode='w') as f:
            f.write(xml)
        print(img_path)
        # cv2.imwrite('temp/{}_shape-image.png'.format(i), image_output)