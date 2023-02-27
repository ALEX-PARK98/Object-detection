import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def another(bgrpic):
    graypic = bgrpic[:, :, 0] * 0.114 + bgrpic[:, :, 1] * 0.547 + bgrpic[:, :, 2] * 0.299
    graypic = np.asarray(graypic, dtype=np.uint8)
    return graypic


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
    graypic = np.asarray(graypic, dtype=np.uint8)
    return graypic

if __name__ == "__main__":
    choonsik = cv.imread("choonsik.jpg")
    convert_choonsik = convert(choonsik)
    cvt_choonsik = cv.cvtColor(choonsik, cv.COLOR_BGR2GRAY)
    another_choonsik = another(choonsik)
    cv.imshow("choonsik.jpg", choonsik)
    cv.imshow("cvt choonsik.jpg", cvt_choonsik)
    cv.imshow("convert choonsik.jpg", convert_choonsik)
    cv.imshow("another choonsik.jpg", another_choonsik)
    cv.waitKey(0)
    cv.destroyAllWindows()
