import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from   skimage.feature import peak_local_max
from   skimage.morphology import watershed
from   scipy import ndimage
import argparse
import imutils
import sys
import math
%matplotlib inline
############################################################
image = cv2.imread("/content/drive/My Drive/Image/circuit.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
# flatten the labels array
labels = labels.flatten()

masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()
import numpy as np
import sys
import cv2 as cv

   
src = masked_image
plt.imshow(src)
plt.show()
    
if len(src.shape) != 2:
  gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
else:
  gray = src

gray = cv.bitwise_not(gray)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 15, -2)

plt.imshow(bw)
plt.show()
    
horizontal = np.copy(bw)
vertical = np.copy(bw)
    
cols = horizontal.shape[1]
horizontal_size = cols // 30

# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    
# Apply morphology operations
horizontal = cv.erode(horizontal, horizontalStructure)
horizontal = cv.dilate(horizontal, horizontalStructure)
    
# Show extracted horizontal lines
plt.imshow(horizontal)
plt.show()
# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows // 30

# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    
# Apply morphology operations
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)
    
plt.imshow(vertical)
plt.show()
 

# Inverse vertical image
vertical = cv.bitwise_not(vertical)

plt.imshow(vertical)
plt.show()

# Step 1
edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 3, -2)

plt.imshow(edges)
plt.show()
 
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv.dilate(edges, kernel)

plt.imshow(edges)
plt.show()

# Step 3
smooth = np.copy(vertical)

# Step 4
smooth = cv.blur(smooth, (2, 2))
    
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]
    
# Show final result
plt.imshow(vertical)
plt.show()

# [smooth]
  

