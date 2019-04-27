import cv2
import numpy as np
import copy

class SheetPreProcessing:

    def __init__(self):
        self.contours_horizon = None
        self.contours_vertical = None

    def find_sheet_contours(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_gray = cv2.threshold(img_gray, 100,  255, cv2.THRESH_BINARY_INV)
        img_gray_horizon = img_gray.copy()
        img_gray_vertical = img_gray.copy()
        structsize = int(img_gray.shape[1]/30)
        hsize = (structsize, 1)
        vsize = (1, structsize)
        structure_horizon = cv2.getStructuringElement(cv2.MORPH_RECT, hsize)
        structure_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, vsize)
        img_gray_horizon = cv2.erode(img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_horizon = cv2.dilate(img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_vertical = cv2.erode(img_gray_vertical, structure_vertical, (-1, -1))
        img_gray_vertical = cv2.dilate(img_gray_vertical, structure_vertical, (-1, -1))
        self.contours_horizon, hierarchy = cv2.findContours(img_gray_horizon, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_vertical, hierarchy = cv2.findContours(img_gray_vertical, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


if __name__ == '__main__':

    test = cv2.imread('test_images/IMG_0788.jpg')
    parameter = SheetPreProcessing()
    parameter.find_sheet_contours(test)
    size = (test.shape[1], test.shape[0])
    output = np.zeros(size)
    cv2.drawContours(test, parameter.contours_horizon, -1, (0, 255, 255), 3)
    cv2.drawContours(test, parameter.contours_vertical, -1, (0, 255, 255), 3)
    output = cv2.resize(test, (int(test.shape[1]/2), int(test.shape[0]/2)))
    cv2.imshow("test", output)
    cv2.waitKey(0)



