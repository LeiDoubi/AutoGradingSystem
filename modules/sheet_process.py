import numpy as np
import cv2 as cv
from . import geometry
import time
import os


class Sheet:
    def __init__(self, imgPath, active_threshold_on=False):
        self.imgPath = imgPath
        # assert(os.path.isabs(imgPath))
        self._activate_threshold_on = active_threshold_on
        self.preprocessImg()
        self.detected_crosses = None

    def preprocessImg(self):
        self.img_original = cv.imread(self.imgPath)
        if self.img_original is None:
            raise FileNotFoundError(
                'Image can\'t be loaded with the path:{}'.format(self.imgPath)
            )
        self.img_gray = cv.cvtColor(self.img_original, cv.COLOR_RGB2GRAY)
        self.img_gray_3channel = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2RGB)
        # remove noise
        self.img_blur = cv.GaussianBlur(self.img_gray, (5, 5), 0)

        # use activate threshold only in necessary case
        if self._activate_threshold_on:
            self.img_bi = cv.adaptiveThreshold(
                self.img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY_INV, 11, 2)
        else:
            _, self.img_bi = cv.threshold(
                self.img_blur, 210, 255, cv.THRESH_BINARY_INV)
        pass


class AnswerSheet(Sheet):
    def __init__(
            self,
            img_path,
            nquestions=33,
            save_result=True,
            dst_dir_path='results/',
            dst_file_name=None,
            *args,
            **kwargs):
        self.nquestions = nquestions
        if dst_file_name is None:
            self.dst_img_path = os.path.join(
                dst_dir_path,
                os.path.basename(img_path))
        else:
            self.dst_img_path = os.path.join(
                dst_dir_path,
                dst_file_name)
        self.save_result = save_result
        self.table_info = {'cell_w': None,
                           'cell_h': None,
                           'left_up_corner': [None, None]}  # w,h
        # the answers of this sheet represented as a numpy array
        self.answers = None
        self.default_map = None
        super().__init__(img_path, *args, **kwargs)

    def _findContours(self):
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
        # find contours and be compatible with different version of opencv
        self._contours, self._hierarchy, *_ = cv.findContours(
            self.result_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    def findRects(self):
        # make sure contours exists
        if not hasattr(self, '_contours'):
            self._findContours()
        # find all rectangles
        self.rects = []
        for idx, contour in enumerate(reversed(self._contours)):
            # for each closed contour, calculate epsilon for approximation
            epsilon = 0.05 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Rectangle has four vertices
                self.rects.append(approx)
        # convert to numpy array

    def mapRects2Table(self):
        '''
            Save the vertices of every found rectangles in ndarry
        '''
        self.table = []
        y_max_err = 9
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
        rects_float = rects.astype('f')
        for idx in range(len(rects)):
            index_sorted = np.argsort(
                (rects_float[idx, :-1, :, 0]**2+rects_float[idx, :-1, :, 1]**2).flatten())
            rects[idx, :-1, :, :] = rects[idx, index_sorted, :, :]
        idx = 0
        idx_1st_Answer_row = 0
        # in case the number of detected cells in first line smaller than 5
        while idx < rects.shape[0]:
            if not np.all(np.abs(rects[idx:idx+3, 4, 0, 1] -
                                 np.mean(rects[idx:idx+3, 4, 0, 1]))
                          < y_max_err):
                idx = idx + 1
            else:
                idx_1st_Answer_row = idx+3
                while np.abs(
                    rects[idx_1st_Answer_row, 4, 0, 1] -
                    np.mean(rects[idx:idx+3, 4, 0, 1])
                ) < y_max_err:
                    idx_1st_Answer_row = idx_1st_Answer_row + 1
                break
        self.table.append(rects[idx:idx_1st_Answer_row, :, :, :])
        # the y of the centroids in the last line
        y_mc_lastLine = np.mean(rects[idx:idx_1st_Answer_row, :, :, 1])
        # use kmeans to estimate the height of cell
        kmeans_criteria = (cv.TERM_CRITERIA_EPS +
                           cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        pos_corners_firstLine = rects[idx:idx_1st_Answer_row, :-1, :, 1].\
            reshape((idx_1st_Answer_row-idx)*4, 1)
        pos_corners_firstLine = pos_corners_firstLine.astype(np.float32)
        _, _, kmeans_centers = cv.kmeans(
            pos_corners_firstLine,
            2,
            None,
            kmeans_criteria,
            3,
            cv.KMEANS_RANDOM_CENTERS
        )
        height_cell = np.abs(kmeans_centers[0]-kmeans_centers[1])
        idx = idx_1st_Answer_row
        while idx+4 < rects.shape[0]:
            ymean_mc_FiveCells = np.mean(rects[idx:idx+5, 4, 0, 1])
            isFiveSuccessiveCellsInLine = np.all(
                np.abs(rects[idx:idx+5, 4, 0, 1] -
                       ymean_mc_FiveCells) < y_max_err)
            isFiveSuccessiveCellsInNextLine = np.abs(
                ymean_mc_FiveCells -
                (y_mc_lastLine+height_cell)) < y_max_err
            if isFiveSuccessiveCellsInNextLine and isFiveSuccessiveCellsInLine:
                # sort the 5 cells from left to right
                # based on the x of their centriods
                sorted_idx = np.argsort(rects[idx:idx+5, 4, :, 0].flatten())
                # append sorted 5 cells aligned in one line
                self.table.append(rects[idx+sorted_idx, :, :, :])
                y_mc_lastLine = ymean_mc_FiveCells
                idx = idx + 5
            else:
                self.table.append(None)
                while idx+4 < rects.shape[0]:
                    # print(rects[idx, 4, :, 1], (y_mc_lastLine+height_cell*2))
                    isCellInNextLine = np.abs(
                        rects[idx, 4, :, 1]-(y_mc_lastLine+height_cell*2)
                    ) < y_max_err
                    if isCellInNextLine:
                        # update the y of centriods in last line
                        y_mc_lastLine = y_mc_lastLine+height_cell
                        break
                    idx = idx + 1

    def calculate_cell_w_h(self):
        self.table_info['cell_w'] = (self.table[0][2:, -1, :, 0] -
                                     self.table[0][1:-1, -1, :, 0]).mean()
        for i in range(1, len(self.table)):
            if self.table[i] is not None:
                self.table_info['cell_h'] = \
                    (self.table[i][:, -1, :, 1].mean() -
                     self.table[0][:, -1, :, 1].mean())/i
        self.table_info['left_up_corner'][0] = \
            self.table[0][0, -1, :, 0] - self.table_info['cell_w']/2
        self.table_info['left_up_corner'][1] = \
            self.table[0][0, -1, :, 1] - self.table_info['cell_h']/2

    def estimate_chopped_lines_center_h(self):
        '''
        return ndarry
            shape:       N*2, 
            1st col:     question Nr
            2nd col:     corresponding vertical ordinate. 
        '''
        ordinate_question = []
        if not hasattr(self, 'calculate_cell_w_h'):
            self.calculate_cell_w_h()
        cell_h = self.table_info['cell_h']
        for idx, cell in enumerate(self.table):
            if cell is None:
                for idx_temp in reversed(range(idx)):
                    if self.table[idx_temp] is not None:
                        h = self.table[idx_temp][0, -1, 0, 1]+\
                            (idx-idx_temp)*cell_h
                        ordinate_question.append(
                            [idx, h])
                        break
            if idx > self.nquestions:
                break
        return np.array(ordinate_question)

    def detectCrosses(self, alpha=0.2):
        '''
        This function detect whether cross exist
        in the each cell within the lines
        corresponding to from 1st question to last question
        '''
        table = self.table
        self.answers = np.zeros((len(table), 4), dtype=bool)
        overlay = self.img_gray_3channel.copy()
        # index of the rows that from the n_question-th row on has content
        self._inds_ContentInRow = set()
        # iterate from 1st to last question

        for i in range(1, len(table)):
            # skip the chopped off rows
            if table[i] is not None:
                # skip the first column
                for j in range(1, 5):
                    iscrossincell, isabnormal, lines, intersections = \
                        geometry.detectCrossinCell(
                            self.img_bi, table[i][j, :-1, :, :])
                    if lines is not None:
                        if isabnormal:
                            linecolor = (255, 0, 0)
                        elif iscrossincell:
                            linecolor = (214, 44, 152)
                            self.answers[i-1, j-1] = 1
                            if i > self.nquestions:
                                self._inds_ContentInRow.add(i)
                        else:
                            linecolor = (20, 255, 20)
                        for line in lines:
                            cv.line(
                                overlay,
                                (line[0, 0], line[0, 1]),
                                (line[0, 2], line[0, 3]),
                                linecolor,
                                2)
                        if intersections is not None:
                            for intersection in intersections:
                                cv.circle(overlay,
                                          (int(intersection[0]), int(
                                              intersection[1])),
                                          2, (0, 0, 255), -1)
        self.img_cross_detected = cv.addWeighted(self.img_gray_3channel,
                                                 1-alpha,
                                                 overlay,
                                                 alpha,
                                                 0)
        if self.save_result:
            cv.imwrite(self.dst_img_path, self.img_cross_detected)
        self._inds_ContentInRow = list(self._inds_ContentInRow)

    def set_default_map(self):
        inds_to_be_corrected_answers = []
        for ind in range(1, 1+self.nquestions):
            if self.table is None:
                inds_to_be_corrected_answers.append(ind)
        # if the lengh of both list are compatible, then set the default map
        if len(self._inds_ContentInRow) == len(inds_to_be_corrected_answers):
            self.default_map = np.array([inds_to_be_corrected_answers,
                                         self._inds_ContentInRow],
                                        dtype=np.int8).T

    def drawTable(self):
        gray_3channel = self.img_gray_3channel.copy()
        # skip the first row
        table = self.table[1:]
        map_idx2char = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D'
        }
        for i, cellsInLine in enumerate(table):
            if cellsInLine is not None:
                for j in range(1, cellsInLine.shape[0]):
                    cv.drawContours(
                        gray_3channel,
                        [cellsInLine[j, :-1, :, :].astype(np.int32)],
                        -1,
                        (255, 0, 0),
                        3
                    )
                    cv.putText(
                        gray_3channel,
                        str(i+1)+','+map_idx2char[j],
                        (cellsInLine[j, 4, :, 0]-20,
                         cellsInLine[j, 4, :, 1]+15),
                        cv.FONT_HERSHEY_PLAIN,
                        2,
                        (20, 20, 255),
                        4
                    )

                    cv.imshow('Find cells', gray_3channel)
                    cv.waitKey(1)
        cv.waitKey(0)

    def drawRect(self, time=200):

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

    def hough_trans_ROI(self, rect):
        import time
        starttime = time.time()
        # rect.shape = (4,1,2)
        i_1 = rect[0, 0, 1]
        i_2 = rect[3, 0, 1]
        j_1 = rect[0, 0, 0]
        j_2 = rect[1, 0, 0]

        lines = cv.HoughLinesP(
            self.img_bi[i_1:i_2+1, j_1:j_2+1], 1, np.pi/20, 7, minLineLength=7)
        lines[:, 0, [0, 2]] = lines[:, 0, [0, 2]]+j_1
        lines[:, :, [1, 3]] = lines[:, :, [1, 3]]+i_1
        elapsed_time = time.time()
        print('needed time: {}'.format((elapsed_time-starttime)*260))
        return lines

    def showImg(self):
        # self.run()
        # self.drawRect()
        # self._findContours()
        cv.namedWindow('IMG', cv.WINDOW_NORMAL)
        cv.imshow('IMG', self.img_bi)
        cv.waitKey()
        cv.destroyAllWindows()

    def run(self):
        starttime = time.time()
        self.findRects()
        self.mapRects2Table()
        self.calculate_cell_w_h()
        # self.drawTable()
        self.detectCrosses()
        self.set_default_map()
        print('needed time:{}s'.format(time.time()-starttime))


class CoverSheet(Sheet):
    pass


def add_map_info(img, img_original, dst_question_number, loc):
    pass


if __name__ == '__main__':
    # testsheet = AnswerSheet('test_images/IMG_0792.jpg')
    # testsheet = AnswerSheet('test_images/IMG_0795.jpg')
    # testsheet = AnswerSheet('test_images/IMG_0797.jpg')
    # testsheet = AnswerSheet('test_images/IMG_0799.jpg')
    # testsheet = AnswerSheet('test_images/IMG_0811.jpg')
    testsheet = AnswerSheet('scan/scan-04.jpg')
    testsheet.run()
    # testsheet.showImg()
