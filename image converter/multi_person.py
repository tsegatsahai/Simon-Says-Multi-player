import cv2
import time
import numpy as np
from random import randint
import argparse


image1 = cv2.imread("three.jpg")
#image1 = cv2.resize(image, (256,256))

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]



def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


frameWidth = image1.shape[1]
frameHeight = image1.shape[0]

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
threshold = 0.1

#the detection goes from (my) right to left
for part in range(nPoints):
    probMap = output[0,part,:,:]
    probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)


#frameClone = image1.copy()
maxPoint1 = detected_keypoints[0][0][0:2]
maxPoint2 = detected_keypoints[0][0][0:2]

frameClone = np.zeros((frameHeight, frameWidth ,3), np.uint8)
for i in range(nPoints):
    #i think j is how many people there are
    for j in range(len(detected_keypoints[i])):
        #for the first person
        if j == 0:
            if detected_keypoints[i][j][0:2] > maxPoint1:
                maxPoint1 = detected_keypoints[i][j][0:2] 
        #for the second person
        if j == 1:
            if detected_keypoints[i][j][0:2] > maxPoint2:
                maxPoint2 = detected_keypoints[i][j][0:2]
        # detected_keypoints[i][j] is where the x,y coordinate is, the [0:2] is splicing (there are
        # 4 elements in the tuple and it only wants the first two numbers
        #cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (255,255,255), -1, cv2.LINE_AA)
        #cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (222,185,90), -1, cv2.LINE_AA)
        
    #print("Detected keypoint at", i, detected_keypoints[i])

#to find the smallest of one person, you need to find the smallest of detected_keypoints[0][j] (you need to compare everything 
#in j...
#cv2.imshow("Keypoints",frameClone)
#cv2.circle(frameClone, maxPoint1, 8, (255,0,0), 2, cv2.LINE_AA)
#cv2.circle(frameClone, maxPoint2, 8, (255,153,255), 2, cv2.LINE_AA)
print("Max keypoint 1 id:", maxPoint1)
print("Max keypoint 2 id:", maxPoint2)
#black image to draw binary skeleton on 
#imgg = np.zeros((frameHeight, frameWidth ,3), np.uint8)




#x_cord = (detected_keypoints[4][0][0] + detected_keypoints[7][1][0]) // 2
#y_cord = (detected_keypoints[4][0][1] + detected_keypoints[7][1][1]) // 2
#cv2.circle(frameClone, (x_cord, y_cord), 8, (255,153,255), 2, cv2.LINE_AA)

print(detected_keypoints[4][0][0])

#major joints are wrists, elbows, and shoulders
#that's what I'm going to check to find the min and max coordinates of the points
majorJoints = [2, 3, 4, 5, 6, 7]
maxOfFirstPerson = -999
indexOfMax = 0
minOfSecondPerson = 999
indexOfMin = 0

#sort the major joints since the keypoints are not generated in order 
#sort them by the x-value 
for i in majorJoints:
    detected_keypoints[i].sort(key=lambda x:x[0])
   
        

#for-loop to find the min of the first person and the max of the second person 
#to be used as coordinates when it's time for cropping the image 
for i in (majorJoints):
    for j in range(len(detected_keypoints[i])):
        if j == 1:
            if detected_keypoints[i][j][0] < minOfSecondPerson:
                minOfSecondPerson = detected_keypoints[i][j][0]
                indexOfMin = i
        elif j == 0:
            if detected_keypoints[i][j][0] > maxOfFirstPerson:
                maxOfFirstPerson = detected_keypoints[i][j][0]
                indexOfMax = i
            
print("max:", maxOfFirstPerson, "index:", indexOfMax)
print("min:", minOfSecondPerson, "index:", indexOfMin)

#for-loop to calculate the x and y coordinates of the middle of the max and min
x_cord = (detected_keypoints[indexOfMin][1][0] + detected_keypoints[indexOfMax][0][0]) // 2
y_cord = (detected_keypoints[indexOfMin][1][1] + detected_keypoints[indexOfMax][0][1]) // 2  
#cv2.circle(frameClone, (x_cord, y_cord), 8, (255,153,200), 2, cv2.LINE_AA)           

print("x_cord:", x_cord)
print("y_cord", y_cord)

print("min of Second person:", end='')
print(detected_keypoints[indexOfMin][0][0:2])

print("max of the first person:", end='')
print(detected_keypoints[indexOfMax][1][0:2])

#cv2.circle(frameClone, (detected_keypoints[indexOfMin][1][0:2]), 8, (255,153,255), 2, cv2.LINE_AA) 
#cv2.circle(frameClone, (detected_keypoints[indexOfMax][0][0:2]), 8, (255,153,255), 2, cv2.LINE_AA)   
#cv2.circle(frameClone, (x_cord, frameHeight), 8, (255,153,200), 2, cv2.LINE_AA) 
#cv2.circle(frameClone, (0, 0), 8, (255,153,200), 2, cv2.LINE_AA)

ankleIndexes = [10, 13]
maxHeight = 0
for i in ankleIndexes:
    for j in range(len(detected_keypoints[i])):
        if detected_keypoints[i][j][1] > maxHeight:
            maxHeight = detected_keypoints[i][j][1]
            
eyeIndexes = [14, 15]
minHeight = maxHeight
for i in eyeIndexes:
    for j in range(len(detected_keypoints[i])):
        if detected_keypoints[i][j][1] < minHeight:
            minHeight = detected_keypoints[i][j][1]
    
print("frameWidth:", frameWidth)
print("frameHeight:", frameHeight)

#cv2.rectangle(frameClone,(0,0),(x_cord,frameHeight),(0,255,0),3)
#newImg = frameClone[(0,minHeight), (x_cord, maxHeight)]
#cv2.imwrite('cropped.jpg', newImg)
  
valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        #cv2.line(imgg, (B[0], A[0]), (B[1], A[1]), (255,255,255), 3, cv2.LINE_AA)
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]),(255,255,255), 3, cv2.LINE_AA)


#cv2.imshow("Detected Pose" , frameClone)
#cv2.imwrite('result.jpg', imgg)

"""
gray = frameClone.copy()
gray = cv2.cvtColor(frameClone, cv2.COLOR_RGB2GRAY)
ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = frameClone[y:y+h,x:x+w]
restOfImage = frameClone[y:y+h, (x):(x)+(frameWidth - (x))]


for i in range(nPoints):
    for j in range(len(detected_keypoints[i])):
        if j == 0:
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (255,153,255), -1, cv2.LINE_AA)
        elif j == 1:
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (0,255,0), -1, cv2.LINE_AA)
        else:
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (185,255,0), -1, cv2.LINE_AA)


restOfImage = frameClone.copy();
gray = restOfImage.copy()
"""

#number of players 
size = len(detected_keypoints[0])
#print("size", size)

#keep this code because you're going to write about it in your paper
"""
starting_x = 0
for j in range(size):
    #if it's the last player, the rest of the image will be used
    if j == size-1:
        #image is the rest of the image
        crop = frameClone[minHeight:minHeight+maxHeight, starting_x:starting_x+(frameWidth - x_cord)]
        cv2.imwrite('player'+str(j)+'.jpg', crop)
    else:
        #cropping the image based on the max and min values of the major joints of each player 
        for i in (majorJoints):
            for k in range(2):
                if k == 0:
                    if detected_keypoints[i][j][0] < minOfSecondPerson:
                        minOfSecondPerson = detected_keypoints[i][j][0]
                        indexOfMin = i
                elif k == 1:
                    if detected_keypoints[i][j+1][0] > maxOfFirstPerson:
                        maxOfFirstPerson = detected_keypoints[i][j+1][0]
                        indexOfMax = i
        x_cord = (detected_keypoints[indexOfMin][j+1][0] + detected_keypoints[indexOfMax][j][0]) // 2 
        
        crop = frameClone[minHeight:minHeight+maxHeight, starting_x:starting_x + (x_cord + (-1*starting_x))]
        cv2.imwrite('player'+str(j)+'.jpg', crop)
        starting_x = x_cord
"""          

            
    
    
    
    
            
for j in range(size):
    #find the max and min joint locations of player j
    #to be used as coordinates when it's time for cropping the image 
    for i in (majorJoints):
        #minimum of player j
        if detected_keypoints[i][j][0] < minOfSecondPerson:
                minOfSecondPerson = detected_keypoints[i][j][0]
                indexOfMin = i
        #maximum of player j
        if detected_keypoints[i][j][0] > maxOfFirstPerson:
                maxOfFirstPerson = detected_keypoints[i][j][0]
                indexOfMax = i

    minn = detected_keypoints[indexOfMin][j][0]
    maxx = detected_keypoints[indexOfMax][j][0]
    crop = frameClone[minHeight:minHeight+(maxHeight-minHeight), minn:minn + (maxx - minn)]
    cv2.imwrite('players/player'+str(j)+'.jpg', crop)



cv2.imwrite('result.jpg', frameClone)






