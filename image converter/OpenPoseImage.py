import cv2
import time
import numpy as np
from PIL import Image
import os
import sys


def convertImage(filePath, outName):
    t = time.time()

    #the .prototxt file speficies the architecture of the neural network
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    #the .caffemodel file stores the weight of the trained model
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    #{Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6...}
    
    imageName = str(outName) + '.jpg'
    frame = cv2.imread(filePath)
    #stores an array copy of the image
    frameCopy = np.copy(frame)
    #print(frameCopy)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    #read the network into memory 
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    
    #converting the image into an input blob so it can be fed into the network 
    #converts image from openCV format to Caffe blob format 
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    #set the inpBlob as the input blob for the network 
    net.setInput(inpBlob)
    
    #passing the image to the model
    output = net.forward()
    
    #output will return a 4D matrix where the first element is the image id,
    #the second element is index of a keypoint
    #the third and fourth are the height and width of the image 

    #gather height and width from the output produced
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
            
    #to draw a binary image, first create a black image
    imgg = np.zeros((frameHeight, frameWidth ,3), np.uint8)
   
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(imgg, points[partA], points[partB], (255, 255, 255), 9)
            #cv2.circle(imgg, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    gray = imgg.copy()
    gray = cv2.cvtColor(imgg, cv2.COLOR_RGB2GRAY)
    ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #image, contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = imgg[y:y+h,x:x+w]

    #Must create this folder first and rename it for each pose
    cv2.imwrite('multi/' + imageName, crop)
    #cv2.imshow('output' , crop)


def getAllImages(rootdir):
    """Gets all of the images in a folder and runs them through OpenPose.
    You just need to run this function with the folder of pose images"""
    path, dirs, files = next(os.walk(rootdir))
    file_count = len(files) - 1
    poseList = []

    i = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                i += 1
                printProgressBar(i + 1, file_count, prefix='Progress:', suffix='Complete', length=50)
                convertImage(filepath,i)

#getAllImages('pose4a')
convertImage('try.jpg', 'try-result')