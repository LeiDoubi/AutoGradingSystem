import cv2
import numpy as np
from modules.sheet_process import AnswerSheet

x1=0
y1=0
flag = 0

def setCallback(img, table, img_orig):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', OnMouseAction)
    while (1):
        cv2.imshow('image', img)
        if(x1>638)and(y1>353):
            calcPosition(100, 33, 638, 353, x1, y1, img, table, img_orig)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


def OnMouseAction(event, x, y, flags, param):
    global x1, y1, flag
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        flag = 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        x1, y1 = x, y
        flag = 2



def calcPosition(width, height, x_0, y_0, x, y, img_gray_3_channel, table, img_orig):

    num_mom_left = int((x - x_0) / width)

    distance_to_mom_left = (x-x_0) % width
    num_mom_upper = int((y-y_0)/height)

    #print(num_mom_upper, y, y_0)
    distance_to_mom_upper = (y-y_0) % height

    if distance_to_mom_upper >= int(height/2)+ 5:
        tsk = num_mom_upper+1
    else:
        tsk = num_mom_upper
    if distance_to_mom_left >= int(width/2)+ 10 :
        aws = num_mom_left+1
    else:
        aws = num_mom_left
   # print(tsk, aws)
    selected_rect = table[tsk][aws, :-1, :, :]
    rect_float = selected_rect.astype('float64')
    index_sorted = np.argsort(
        (rect_float[:, :, 0] ** 2 + rect_float[:, :, 1] ** 2).flatten())
    leftupperCorn = selected_rect[index_sorted[0], :, :]
    rightlowerCorn = selected_rect[index_sorted[-1], :, :]
    leftupper = (leftupperCorn[0,0], leftupperCorn[0,1])
    rightbottom= (rightlowerCorn[0,0], rightlowerCorn[0,1])

    if flag == 1:
        if (tsk !=0) and (aws !=0):
            for i in range(min(leftupperCorn[0,0], rightlowerCorn[0,0]), max(leftupperCorn[0,0], rightlowerCorn[0,0])):
                for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):
                    img_gray_3_channel[j, i] = img_orig[j, i]
                    img_gray_3_channel[j,i,0]=img_gray_3_channel[j,i,0]*0.8+0*0.2
                    img_gray_3_channel[j, i, 1] = img_gray_3_channel[j, i, 1] * 0.8 + 0 * 0.2
                    img_gray_3_channel[j, i, 2] = img_gray_3_channel[j, i, 2] * 0.8 + 255 * 0.2
    elif flag == 2:
        if (tsk !=0) and (aws !=0):
            for i in range(min(leftupperCorn[0,0], rightlowerCorn[0,0]), max(leftupperCorn[0,0], rightlowerCorn[0,0])):
                for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):
                    img_gray_3_channel[j, i] = img_orig[j, i]

def tskmapping(crossedoutlist):






if __name__ == '__main__':
    test = cv2.imread('../results/scan-02.jpg')
    answer_sheet = AnswerSheet('../scan/scan-02.jpg')
    answer_sheet.run()

    setCallback(test, answer_sheet.table, answer_sheet.img_original)
