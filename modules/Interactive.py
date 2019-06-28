import cv2
import numpy as np
from modules.sheet_process import AnswerSheet

x1=0
y1=0
flag = 0
mode = 0
tomaptsk = 0

def setCallback(img, table, img_orig, ordinate_questions):
    global mode
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', OnMouseAction)
    while (1):
        cv2.imshow('image', img)

        if(mode ==0):
            if(x1>705)and(y1>353):
                calcPosition(100, 33, 638, 353, x1, y1, img, table, img_orig)
        if (mode == 1):
               if (x1<705) and (x1 > 573) and (y1>353):
                    tskmapping(ordinate_questions, table, x1, y1, 353, 33, img, img_orig)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('e'):
            mode =1 - mode
            if mode == 1:
                print('Edit mode')
            elif(mode == 0):
                print('quit edit mode')
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


    distance_to_mom_upper = (y-y_0) % height

    if distance_to_mom_upper >= int(height/2)+ 5:
        tsk = num_mom_upper+1
    else:
        tsk = num_mom_upper
    if distance_to_mom_left >= int(width/2)+ 10 :
        aws = num_mom_left+1
    else:
        aws = num_mom_left
    if table[tsk] is not None:
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

def tskmapping(ordinate_question,table, x,  y, y_0, height, img_gray_3channel, img_orig):
    global tomaptsk, tomaptsk_y, tomaptsk_x

    num_mom_upper = int((y - y_0) / height)
    distance_to_mom_upper = (y - y_0) % height

    if distance_to_mom_upper >= int(height / 2) + 5:
        tsk = num_mom_upper + 1
    else:
        tsk = num_mom_upper
    if tsk > 33:
        tomaptsk = tsk
        tomaptsk_y = y
        tomaptsk_x = x
    tsk_height = ordinate_question[:,1]
    if table[tomaptsk] is not None:
        selected_rect = table[tomaptsk][0, :-1, :, :]
        rect_float = selected_rect.astype('float64')
        index_sorted = np.argsort(
            (rect_float[:, :, 0] ** 2 + rect_float[:, :, 1] ** 2).flatten())
        leftupperCorn = selected_rect[index_sorted[0], :, :]
        rightlowerCorn = selected_rect[index_sorted[-1], :, :]

    idx = (np.abs(tsk_height - y)).argmin()
    if flag == 1:
        if tsk == ordinate_question[idx][0] and tomaptsk !=0:

            cv2.putText(
                img_gray_3channel,
                str(tsk) + '<-',
                (500,int((min(leftupperCorn[0,1], rightlowerCorn[0,1])+max(leftupperCorn[0,1], rightlowerCorn[0,1]))/2)+8),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (20, 20, 255),
                4
            )
    elif flag == 2:
        for i in range(500, max(leftupperCorn[0,0], rightlowerCorn[0,0])):
            for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):
                img_gray_3channel[j, i] = img_orig[j, i]










if __name__ == '__main__':
    test = cv2.imread('../results/scan-02.jpg')
    answer_sheet = AnswerSheet('../scan/scan-02.jpg')
    answer_sheet.run()
    setCallback(test, answer_sheet.table, answer_sheet.img_original, answer_sheet.estimate_chopped_lines_center_h())
