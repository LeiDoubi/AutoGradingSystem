import cv2 as cv
import numpy as np


class Sheet:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        # assert(os.path.isabs(imgPath))
        self._processImg()

    def _processImg(self):
        self.img_original = cv.imread(self.imgPath)
        if self.img_original is None:
            raise FileNotFoundError(
                'Image can\'t be loaded with the path:{}'.format(self.imgPath)
            )
        self.img_gray = cv.cvtColor(self.img_original, cv.COLOR_RGB2GRAY)
        # remove noise
        img_blur = cv.GaussianBlur(self.img_gray, (5, 5), 0)

        # use a fixed threshold to get binary img
        retval, self.img_bi = cv.threshold(
            img_blur, 100, 255, cv.THRESH_BINARY_INV)
        pass


class AnswerSheet(Sheet):
    def findContours(self):
        # set morphology structures size
        structsize = int(self.img_gray.shape[1]/30)

        # hsize for horizontal lines, vsize for vertical lines
        hsize = (structsize, 1)
        vsize = (1, structsize)
        # morphology structures
        structure_horizon = cv.getStructuringElement(cv.MORPH_RECT, hsize)
        structure_vertical = cv.getStructuringElement(cv.MORPH_RECT, vsize)

        # erode and dilate images horizontal and vertical
        img_gray_horizon = cv.erode(
            self.img_bi.copy(), structure_horizon, (-1, -1))
        img_gray_horizon = cv.dilate(
            img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_vertical = cv.erode(
            self.img_bi.copy(), structure_vertical, (-1, -1))
        img_gray_vertical = cv.dilate(
            img_gray_vertical, structure_vertical, (-1, -1))
        result_binary = cv.addWeighted(
            img_gray_horizon, 1, img_gray_vertical, 1, 0)
        pass
        # find contours
        self._contours, self._hierarchy = cv.findContours(
            result_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        pass

    def findRects(self):
        # make sure contours exists
        if not hasattr(self, '_contours'):
            self.findContours()
        # find all rectangles
        self.rects = []
        for idx, contour in enumerate(reversed(self._contours[1:])):
            # for each closed contour, calculate epsilon for approximation
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Rectangle has four vertices
                self.rects.append(approx)

    def mapRect2Array(self):
        '''
            Save the vertices of every found rectangles in ndarry
        '''
        if not hasattr(self, 'rects'):
            self.findRects()
        # convert to numpy array
        rects = np.array(self.rects)
        # approximately calculate the height of all found rectangles
        # TODO test failed
        sums_posx_posy = rects.sum(axis=3)
        sums_height_weight = sums_posx_posy.max(
            axis=1)-sums_posx_posy.min(axis=1)

        rect_heights = np.vstack((rects[:, 2, 0, 1]-rects[:, 1, 0, 1],
                                  rects[:, 3, 0, 1]-rects[:, 0, 0, 1])).min(axis=0).flatten()
        mask = rect_heights > rect_heights.max()*2/3
        pass
        # filter wo

    # mask = rect_heights > cell_height-

    def drawRect(self, time=1):
        if not hasattr(self, 'rects'):
            self.findRects()
        gray_3channel = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2RGB)
        for _, rect in enumerate(self.rects):
            mm = cv.moments(rect)
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            cv.circle(gray_3channel, (cx, cy), 10, (0, 0, 255), -1)
            cv.drawContours(gray_3channel, [rect], -1, (255, 0, 0), 3)
            cv.imshow('findRects', gray_3channel)
            if _ < len(self.rects)-1:
                cv.waitKey(time)
            else:
                cv.waitKey(0)


class CoverSheet(Sheet):
    pass


if __name__ == '__main__':
    testsheet = AnswerSheet('test_images/IMG_0792.jpg')
    # testsheet.MapRect2Array()
    testsheet.drawRect()
    pass
