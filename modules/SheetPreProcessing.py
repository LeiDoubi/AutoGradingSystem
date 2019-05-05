import cv2
import numpy as np
import copy

class SheetPreProcessing:

    def __init__(self):
        self.contours = None
        self.moment = []
        self.sheet_lines = []


    def find_sheet_contours(self, img):
        #convert img to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # convert grayimg to binary
        ret, img_gray = cv2.threshold(img_gray, 100,  255, cv2.THRESH_BINARY_INV)

        img_gray_horizon = img_gray.copy()
        img_gray_vertical = img_gray.copy()
        # set morphology structures size
        structsize = int(img_gray.shape[1]/40)

        #hsize for horizontal lines, vsize for vertical lines
        hsize = (structsize, 1)
        vsize = (1, structsize)

        # morphology structures
        structure_horizon = cv2.getStructuringElement(cv2.MORPH_RECT, hsize)
        structure_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, vsize)

        #erode and dilate images horizontal and vertical
        img_gray_horizon = cv2.erode(img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_horizon = cv2.dilate(img_gray_horizon, structure_horizon, (-1, -1))
        img_gray_vertical = cv2.erode(img_gray_vertical, structure_vertical, (-1, -1))
        img_gray_vertical = cv2.dilate(img_gray_vertical, structure_vertical, (-1, -1))
        result_binary = cv2.addWeighted(img_gray_horizon, 1, img_gray_vertical, 1, 0)


        #find contours
        self.contours, hierarchy = cv2.findContours(result_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)



        #size for a zero Mat
        size = (img.shape[0], img.shape[1])
        result = np.zeros(size)
        cv2.drawContours(result, self.contours, -1, 255, 3)
        return result


    def findsquares(self, img_contour):

        for i in range(len(self.contours)):
            # for each closed contour, calculate epsilon for approximation
            epsilon = 0.01 * cv2.arcLength(self.contours[i], True)
            approx = cv2.approxPolyDP(self.contours[i], epsilon, True)
            corners = len(approx)
            # find out rectangles approximately
            if corners == 4:
                # calculate moment for a better look                      TO BE DELETED
                mm = cv2.moments(self.contours[i])
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
                cv2.circle(img_contour, (cx, cy), 10, 255, -1)
                self.moment.append((cx, cy))

    def rectfilter(self):
        line = 0

        for i in range(len(self.moment)-5):
            temp = []
            test=[]
            idx=[]
            for t in range(0, 5):
                temp.append(self.moment[i+t][1])
                array_y = np.array(temp)

            mean = array_y.sum()/5
            for t in range(len(array_y)):
                if array_y[t]- mean < 6 and array_y[t] - mean > -6:
                    test.append(array_y[t])

            if len(test) < 5:
                if line == 0 or line == 50:
                    continue

            else:
                self.sheet_lines.append(test)
                line += 1

        for t in range(len(self.sheet_lines)-1):

            tobedeleted = sum(self.sheet_lines[t])/5-sum(self.sheet_lines[t+1])/5
            if sum(self.sheet_lines[t])/5-sum(self.sheet_lines[t+1])/5 >70:
                #print(tobedeleted)
                print(t+1)



















if __name__ == '__main__':
    # read img
    test = cv2.imread('test_images/IMG_0790.jpg')
    parameter = SheetPreProcessing()
    result = parameter.find_sheet_contours(test)
    parameter.findsquares(result)
    parameter.rectfilter()
    #resize for a better look
    output = cv2.resize(result, (int(test.shape[1]/8), int(test.shape[0]/8)))

    cv2.imshow("test", output)
    cv2.waitKey(0)
    #lines = cv2.HoughLinesP(ROI, 1, np.pi / 20, 7, minLineLength=7)
    #cv2.imshow("test", output)
   # cv2.waitKey(0)



