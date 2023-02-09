import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def convert(bgrpic: np.ndarray):
    w = bgrpic.shape[0]
    h = bgrpic.shape[1]
    blue = np.full((w,1), 0.114)
    green = np.full((w,1), 0.547)
    red = np.full((w,1), 0.299)
    gray = np.concatenate((blue, green, red), axis = 1)
    gray = np.reshape(gray, (w,3,1))
    graypic = np.matmul(bgrpic, gray)
    graypic =graypic.flatten()
    graypic = np.reshape(graypic, (w, h))
    graypic = graypic/256
    return graypic

if __name__ == "__main__":
    choonsik = cv.imread("template.jpg")
    gray_choonsik = convert(choonsik)
    gray_choonsik = gray_choonsik
    cvt_choonsik = cv.cvtColor(choonsik, cv.COLOR_BGR2GRAY)
    cvt_choonsik = cvt_choonsik
    cv.imshow("cvt choonsik.jpg", choonsik)
    cv.imshow("cvt choonsik.jpg", cvt_choonsik)
    cv.imshow("gray choonsik.jpg", gray_choonsik)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(cvt_choonsik)
    print(gray_choonsik)
