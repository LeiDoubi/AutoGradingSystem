import cv2 as cv
import numpy as np


class Sheet:
    def __init__(self, imgPath, active_threshold_on=False):
        self.imgPath = imgPath
        # assert(os.path.isabs(imgPath))
        self._activate_threshold_on = active_threshold_on
        self.preprocessImg()

    def preprocessImg(self):
        self.img_original = cv.imread(self.imgPath)
        if self.img_original is None:
            raise FileNotFoundError(
                'Image can\'t be loaded with the path:{}'.format(self.imgPath)
            )
        self.img_gray = cv.cvtColor(self.img_original, cv.COLOR_RGB2GRAY)
        # remove noise
        img_blur = cv.GaussianBlur(self.img_gray, (5, 5), 0)

        # use activate threshold only in necessary case
        if self._activate_threshold_on:
            self.img_bi = cv.adaptiveThreshold(
                img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY_INV, 11, 2)
        else:
            _, self.img_bi = cv.threshold(
                img_blur, 100, 255, cv.THRESH_BINARY_INV)
        pass


class AnswerSheet(Sheet):
    def findContours(self):
        # set morphology structures size
        structsize = int(self.img_gray.shape[1]/70)

        # hsize for horizontal lines, vsize for vertical lines
        hsize = (structsize, 1)
        vsize = (1, structsize)
        # morphology structures
        structure_horizon = cv.getStructuringElement(cv.MORPH_RECT, hsize)
        structure_vertical = cv.getStructuringElement(cv.MORPH_RECT, vsize)

        # erode and dilate images horizontal and vertical
        img_gray_horizon = cv.erode(
            self.img_bi, structure_horizon, (-1, -1))
        img_gray_horizon = cv.dilate(
            img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_vertical = cv.erode(
            self.img_bi, structure_vertical, (-1, -1))
        img_gray_vertical = cv.dilate(
            img_gray_vertical, structure_vertical, (-1, -1))
        self.result_binary = cv.addWeighted(
            img_gray_horizon, 1, img_gray_vertical, 1, 0)
        pass
        # find contours
        self._contours, self._hierarchy = cv.findContours(
            self.result_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        pass

    def findRects(self):
        # make sure contours exists
        if not hasattr(self, '_contours'):
            self.findContours()
        # find all rectangles
        self.rects = []
        for idx, contour in enumerate(reversed(self._contours)):
            # for each closed contour, calculate epsilon for approximation
            epsilon = 0.05 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Rectangle has four vertices
                self.rects.append(approx)
        # convert to numpy array

    def mapRect2Table(self):
        '''
            Save the vertices of every found rectangles in ndarry
        '''
        if not hasattr(self, 'rects'):
            self.findRects()
        # mass center of cells
        mc = np.zeros((len(self.rects), 1, 1, 2), dtype=np.int16)
        for idx, rect in enumerate(self.rects):
            mm = cv.moments(rect)
            mc[idx, 0, 0, 0] = int(mm['m10'] / mm['m00'])
            mc[idx, 0, 0, 1] = int(mm['m01'] / mm['m00'])
        # rects.shape = (N,5,1,2)
        # 1st dim is number of found rectangles
        # 2nd dim 4 vertices and 1 mass center
        # last dim posx, posy
        rects = np.concatenate((np.array(self.rects, dtype=np.int16),
                                mc), axis=1)
        # index of current cell
        idx_current_cell = 0
        for idx in range(rects.shape[0]):
            if np.abs(rects[idx, 5, 0, 1]-rects[idx+1, 5, 0, 1]) < 20:
                idx_current_cell = idx
                break
        colum_height = 
        while idx_current_cell < rects.shape[0]:
            if idx_current_cell+5 < rects.shape[0]:


            # # approximately calculate the height of all found rectangles
            # # TODO test failed
            # sums_posx_posy = rects.sum(axis=3)
            # sums_height_weight = sums_posx_posy.max(
            #     axis=1)-sums_posx_posy.min(axis=1)

            # rect_heights = np.vstack((rects[:, 2, 0, 1]-rects[:, 1, 0, 1],
            #                           rects[:, 3, 0, 1]-rects[:, 0, 0, 1])).\
            #     min(axis=0).flatten()
            # mask = rect_heights > rect_heights.max()*2/3
            # pass
            # # filter wo

            # mask = rect_heights > cell_height-

    def drawRect(self, time=1):

        if not hasattr(self, 'rects'):
            self.findRects()
        cv.imshow('Detected Table', self.result_binary)
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
    testsheet = AnswerSheet('test_images/IMG_0788.jpg')
    testsheet.mapRect2Table()
    # testsheet.drawRect()
    pass
