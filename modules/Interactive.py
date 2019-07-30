import cv2
import numpy as np
from modules.sheet_process import AnswerSheet

x1=0
y1=0
flag = 0
mode = 0
tomaptsk = 0
x_temp = 0
y_temp = 0


def setCallback(img, table, img_orig, ordinate_questions,  tskmap, solutionmatrix):
    global mode, x_temp, y_temp
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', OnMouseAction)
    map_result = None
    solution  = solutionmatrix
    while (1):
        cv2.imshow('image', img)
        if(x_temp!=x1) or (y_temp!=y1):
            x_temp = x1
            y_temp = y1
            if(mode ==0):
                if(x1>705)and(y1>353) and (x1<1085) and (y1<2045):
                   solution = calcPosition(100, 33, 638, 353, x1, y1, img, table, img_orig, solutionmatrix)
            if (mode == 1):
                   if (x1<705) and (x1 > 573) and (y1>353) and (y1<2045):
                    map_result =  tskmapping(ordinate_questions, table, x1, y1, 353, 33, img, img_orig, tskmap)
        k = cv2.waitKey(1)
        if k == ord('n'):
            break
        if k == ord('e'):
            mode =1 - mode
            if mode == 1:
                print('Edit mode')
            elif(mode == 0):
                print('quit edit mode')
    cv2.destroyAllWindows()

    return solution, map_result, img

def OnMouseAction(event, x, y, flags, param):   # mouse action for anwser sheet
    global x1, y1, flag
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        flag = 1
        print(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        x1, y1 = x, y
        flag = 2


def on_mouse(event, x, y, flags, param):   #mouse action for  cover sheet
    global img, point1, point2, g_rect
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:             #point 1
        point1 = (x, y)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):        #drag mouse
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:                      # point 2
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect = [min_x, min_y, width, height]
            cut_img = img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('ROI', cut_img)



def calcPosition(width, height, x_0, y_0, x, y, img_gray_3_channel, table, img_orig, solutionmatrix):
    print(x, y)

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
    x = 0
    y = 0
    if table[tsk] is not None:
        selected_rect = table[tsk][aws, :-1, :, :]
        rect_float = selected_rect.astype('float64')
        index_sorted = np.argsort(
            (rect_float[:, :, 0] ** 2 + rect_float[:, :, 1] ** 2).flatten())
        leftupperCorn = selected_rect[index_sorted[0], :, :]
        rightlowerCorn = selected_rect[index_sorted[-1], :, :]


        if flag == 1:
            if (tsk !=0) and (aws !=0):
                solutionmatrix[tsk-1][aws-1]=True
                for i in range(min(leftupperCorn[0,0], rightlowerCorn[0,0]), max(leftupperCorn[0,0], rightlowerCorn[0,0])):
                    for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):
                        img_gray_3_channel[j, i] = img_orig[j, i]
                        img_gray_3_channel[j,i,0]=img_gray_3_channel[j,i,0]*0.8+0*0.2
                        img_gray_3_channel[j, i, 1] = img_gray_3_channel[j, i, 1] * 0.8 + 0 * 0.2
                        img_gray_3_channel[j, i, 2] = img_gray_3_channel[j, i, 2] * 0.8 + 255 * 0.2
        elif flag == 2:
            if (tsk !=0) and (aws !=0):
                solutionmatrix[tsk - 1][aws - 1] = False
                for i in range(min(leftupperCorn[0,0], rightlowerCorn[0,0]), max(leftupperCorn[0,0], rightlowerCorn[0,0])):
                    for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):
                        img_gray_3_channel[j, i] = img_orig[j, i]
    return solutionmatrix

def tskmapping(ordinate_question,table, x,  y, y_0, height, img_gray_3channel, img_orig, tskmap):
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
        if tsk == ordinate_question[idx][0] and tomaptsk !=0 :
            a = list(( tsk, tomaptsk,))
            if not a in tskmap:
                tskmap.append(a)

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
        if tskmap is not None:
            for i in range(-1, len(tskmap)-1):
                if tskmap[i][1] == tomaptsk:
                    del tskmap[i]

        for i in range(500, max(leftupperCorn[0,0], rightlowerCorn[0,0])):
            for j in range(min(leftupperCorn[0,1], rightlowerCorn[0,1]), max(leftupperCorn[0,1], rightlowerCorn[0,1])):

                img_gray_3channel[j, i] = img_orig[j, i]

    return tskmap


def selectROI(imgPath):
    global img
    img = cv2.imread(imgPath)
    if img is None:
        raise FileNotFoundError(
            'Cover sheet can\'t be loaded with the path:{}'.format(imgPath)
        )
    g_rect = get_image_roi(img)
    return g_rect[0], g_rect[1], g_rect[2], g_rect[3]


def get_image_roi(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while True:
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 13:
            break
    cv2.destroyAllWindows()
    return g_rect









if __name__ == '__main__':
    # image_path="../dataset/images/IMG_0007.JPG"
    image_path = "../scan/scan-01.jpg"

    x, y, w, h = selectROI(image_path)
    print(x, y, w, h)

