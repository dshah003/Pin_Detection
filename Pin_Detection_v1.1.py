import cv2
import numpy as np;
from matplotlib import pyplot as plt
 
# Read image
img = cv2.imread("TestImgResized.jpg", 1)
img = cv2.medianBlur(img,9)    #creates trouble for Green detection. But helpful for red/blue
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

yellowMin = (20,100,100)
yellowMax = (30, 255, 255)

blueMin= (66,24,0)
blueMax= (162,66,63)

redMin=(150,177,78)
redMax=(252,255,231)

greenMin = (50, 200, 0)
greenMax = (100, 255, 250)

red=cv2.inRange(hsv, redMin, redMax)
yellow=cv2.inRange(hsv, yellowMin, yellowMax)
blue=cv2.inRange(img, blueMin, blueMax)
green = cv2.inRange(hsv, greenMin, greenMax)

kernel = np.ones((5,5),np.uint8)
red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)

params = cv2.SimpleBlobDetector_Params()
params.minArea=50
detector = cv2.SimpleBlobDetector_create(params)

# Detect yellow pins.
keypoints = detector.detect(255-yellow)
img1= cv2.drawKeypoints(img, keypoints, np.array([]), (30,250,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Yellow", img1)
print("No. of yellow Pins = " + str(len(keypoints)))

#Detect Blue pins
params.filterByConvexity = 'true';
params.minConvexity = 0.2;
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(255-blue)
img2= cv2.drawKeypoints(img1, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Blue", img_blue)
print("No. of Blue Pins = " + str(len(keypoints)))

#Detect Red Pins
keypoints = detector.detect(255-red)
img3= cv2.drawKeypoints(img2, keypoints, np.array([]), (30,250,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Red", img_red)
print("No. of Red Pins = " + str(len(keypoints)))

#Detect Green Pins
keypoints = detector.detect(255-green)
img4= cv2.drawKeypoints(img3, keypoints, np.array([]), (30,250,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Detected Pins", img4)
print("No. of Green Pins = " + str(len(keypoints)))


cv2.waitKey(0)
