from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os
import numpy as np
import argparse
import imutils
import cv2
import time
import skimage as sk

# from skimage.morphology import skeletonize
# from skimage import data
# import matplotlib.pyplot as plt
# from skimage.util import invert

def prev_dir(filename):
    a = filename.split("/")
    s = ""
    for i in range (len(a)-1):
        s = s + a[i] + "/"
    return s
scriptpath = os.path.realpath(__file__)
dirpath = prev_dir(scriptpath)
# ---------------------------------------
def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    return skel
# ---------------------------------------
def check_active(img,x1,y1):
    # print("Hii")
    if(x1 >= 0 and x1 < 1080 and y1 >= 0 and y1 < 1920):
        # print("Hii")
        # print(img[x1][y1])
        if(img[x1][y1] > 0):
            # print(img[x1][y1])
            return True
    return False
# -----------------------------------------
def local_check(img,x1,y1):
    ctr = 0
    # print(str(x1) + " " + str(y1))
    for i in range(-5,6):
        for j in range(-5,6):
            if(check_active(img,x1+i,y1+j)):
                ctr += 1
    if(ctr >= 1):
        return True
    else:
        return False
# -----------------------------------------
def print_pos(img):
    for i in range(0,1080):
        for j in range(0,1920):
            if(img[i][j] > 0):
                print(str(i) + " " + str(j) + " " + str(img[i][j]))
# -----------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
# ---------------------------------------
inputvideo = dirpath+"Videos/" + args["video"]+".mp4"
vidcap = cv2.VideoCapture(inputvideo)
fps = vidcap.get(cv2.CAP_PROP_FPS)

print(dirpath+"Videos/" + args["video"]+".mp4")
success,image = vidcap.read()
height, width, layers = image.shape
size = (width,height)
print(size)
print("------------------")
count = 0
pathOut = dirpath + "Videos/Output/" + args["video"] + "_out.avi"
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
# ---------------------------------------
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# count = 1
kernel_close = np.ones((15,15),np.uint8)
kernel_open = np.ones((4,4),np.uint8)
kernel = np.ones((3,3),np.uint8)
large = np.ones((50,50),np.uint8)
medium = np.ones((7,7),np.uint8)

while success:
    img = fgbg.apply(image)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, large)
    img = cv2.GaussianBlur(img, (21, 21), 0)

    img = skeletonize(img)
    img = cv2.dilate(img, medium, iterations=1)
    
    horizontalsize = 1920 // 150
    horizontalStructure= cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))
    hrz = cv2.erode(img,horizontalStructure)
    hrz = cv2.dilate(hrz,horizontalStructure)
    hrz = cv2.bitwise_not(hrz)
    img = cv2.bitwise_and(img,hrz)

    img = cv2.dilate(img, medium, iterations=1)
    lines = cv2.HoughLines(img,1,np.pi/180, 100)
    if not (lines is None):
        for r,theta in lines[0]: 
            
            a = np.cos(theta) 
            b = np.sin(theta) 
            x0 = a*r 
            y0 = b*r
            p = 0
            m = 0
            
            plus = 0
            x1 = int(x0 + plus*(-b)) 
            y1 = int(y0 + plus*(a))
            while((not (local_check(img,y1,x1))) and plus < 2000):
                plus = plus + 1
                # print(plus)
                x1 = int(x0 + plus*(-b)) 
                y1 = int(y0 + plus*(a))
            x1_ = x1
            y1_ = y1
            while(local_check(img,y1,x1) and plus < 2000):
                plus = plus + 1
                p = 1
                x1 = int(x0 + plus*(-b))
                y1 = int(y0 + plus*(a))

            minus = -1
            x2 = int(x0 + minus*(-b)) 
            y2 = int(y0 + minus*(a))
            while((not (local_check(img,y2,x2))) and minus > -2000):
                minus = minus - 1
                # print(minus)
                x2 = int(x0 + minus*(-b)) 
                y2 = int(y0 + minus*(a))
            x2_ = x2
            y2_ = y2
            while(local_check(img,y2,x2) and minus > -2000):
                minus = minus - 1
                m = 1
                x2 = int(x0 + minus*(-b)) 
                y2 = int(y0 + minus*(a))

            if(p == 1):
                cv2.line(image,(x1,y1), (x1_,y1_), (247,0,206),5)
            if(m == 1):
                cv2.line(image,(x2,y2), (x2_,y2_), (247,0,206),5)

    # output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # cv2.imshow('frame',output)
    # cv2.waitKey(0)
    out.write(image)
    success,image = vidcap.read()
    # count = count + 1
vidcap.release()
out.release()
cv2.destroyAllWindows()

