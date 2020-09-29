import numpy as np
import cv2

# To run this code, set live to true if you want live SIFT tracking and false if you want to enter an image path
# or capture a frame from video.some issue: video feed still shows after "i" is pressed to capture screenshot. 
# Sometimes SIFT runs continuosly even on static images. Q only quits if live.


########## SETUP VARIABLES##########
#bool to toggle live or still frame mode
LIVE = 1
#FLANN or Brute Force
BF = 0
# switch between image capture and local
LOCAL = 0
#camera
cap = cv2.VideoCapture(0)
#paths
path1 = 'Path/to/img1.jpg'
path2 = 'Path/to/img2.jpg'




#shows video feed until i key is pressed, then uses current frame as img2
def get_frame():
    while True:
        x = cv2.waitKey(20)
        c = chr(x & 0xFF)
        cv2.waitKey(10)
        ret, frame = cap.read()
        cv2.imshow("live", frame)
        if c == "i":
            break
    return frame

#runs SIFT on the video feed
def SIFT_live(img):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors  of static image with SIFT
    kp2, des2 = sift.detectAndCompute(img,None)
    return kp2, des2

#runs SIFT on the static image for reference
def SIFT_static(img):
    img1 = img
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors  of static image with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    return img1, kp1, des1

#runs brute force matcher
def bf(des1, des2):
        brute = cv2.BFMatcher()
        matches = brute.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good

#runs FLANN matcher (based heavily on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
def FLANN(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
    return matches, draw_params

#takes keypoints, descriptors, and images and creates a matched image for BF or FLANN
def match(des1, des2, img1, kp1, img2, kp2):
    if BF:
        good = bf(des1, des2)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        matches, draw_params = FLANN(des1, des2)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

def Bound(good_matches,img_matches,kp1,kp2,img1):
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = kp1[good_matches[i].queryIdx].pt[0]
        obj[i,1] = kp1[good_matches[i].queryIdx].pt[1]
        scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
        scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]
    H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img1.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img1.shape[1]
    obj_corners[2,0,1] = img1.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img1.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv2.line(img_matches, (int(scene_corners[0,0,0] + img1.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img1.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[1,0,0] + img1.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img1.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[2,0,0] + img1.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img1.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[3,0,0] + img1.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img1.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    #-- Show detected matches
    cv2.imshow('Good Matches & Object detection', img_matches)

def main():
    
    #run SIFT on static image
    img1, kp1, des1 = SIFT_static(cv2.imread(path1, cv2.IMREAD_GRAYSCALE))        # queryImage
    
    count = 0
    #loop to continue video feed
    while True:
        
        if LIVE:
            ret, img2 = cap.read()
        else:
            if LOCAL:
                img2 = cv2.imread(path2)
            else:
                img2 = get_frame()


        x = cv2.waitKey(20)
        c = chr(x & 0xFF)


        #convert image to grayscale
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        #run SIFT on live feed
        kp2, des2 = SIFT_live(gray)

        #run matcher
        count += 1
        img3 = match(des1, des2, img1, kp1, img2, kp2)
       
        #show matches
        cv2.imshow("match", img3)
        
        #attempting to make loop stop if not live
        if LIVE == False:
            cv2.waitKey(10)
            #time.sleep(100)

        if c == "q": 
                break

    cv2.destroyAllWindows()
    cv2.waitKey()
    cap.release()


if __name__ == "__main__":
    main()