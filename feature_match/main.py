import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('choonsik.png', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('template.png', cv.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()
kp2, des2 = orb.detectAndCompute(img2, None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # find the key points and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(frame, None)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv.drawMatches(frame, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    if cv.waitKey(1) == ord('q'):
        break

    cap.release()
    cv.destroyAllWindows()
