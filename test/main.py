import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def another(bgrpic):
    graypic = bgrpic[:, :, 0] * 0.114 + bgrpic[:, :, 1] * 0.547 + bgrpic[:, :, 2] * 0.299
    graypic = np.asarray(graypic, dtype=np.uint8)
    return graypic


cap = cv.VideoCapture(0)
#if not cap.isOpened():
#    print("Cannot open camera")
#    exit()

temp = cv.imread("template.jpg", 0)
w, h = temp.shape[::-1]
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    '''
    print(frame.shape) -> (480,640,3)
    print(copyframe.shape) -> (480, 640)
    '''
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    copyframe = another(frame)
    res = cv.matchTemplate(copyframe, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(copyframe, top_left, bottom_right, 255, 2)
    # Our operations on the frame come here

    # Display the resulting frame
    cv.imshow('frame', copyframe)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
