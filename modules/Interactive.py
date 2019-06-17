import cv2
import numpy as np

x1=0
y1=0

def setCallback(img):
    cv2.namedWindow('image')
    #cv2.setMosueCallback('Interactive Window', OnMouseAction)
    cv2.setMouseCallback('image', OnMouseAction)
    while (1):
        cv2.imshow('image', img)
        calcPosition(90, 30, 638, 347, x1, y1, img)
        k = cv2.waitKey(1)
        print(x1, y1)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


def OnMouseAction(event, x, y, flags, param):
    global x1, y1
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        #print(id(x1), y1)



def calcPosition(width, height, x_0, y_0, x, y, img_gray_3_channel):

    num_mom_left = int((x - x_0) / width)+1
    distance_to_mom_left = (x-x_0) % width
    num_mom_upper = int((y-y_0)/height)+1
    distance_to_mom_upper = (y-y_0) % height
    if distance_to_mom_upper >= int(height/2):
        tsk = num_mom_upper+1
    else:
        tsk = num_mom_upper
    if distance_to_mom_left >= int(width/2):
        aws = num_mom_left+1
    else:
        aws = num_mom_left
    #return tsk, aws

    upperleft = (x_0+ aws * width, y_0+ tsk * height)
    bottomright = (x_0 + aws * width+width, y_0 + tsk * height+height)
    cv2.rectangle(img_gray_3_channel, upperleft, bottomright, (0, 255, 255))


if __name__ == '__main__':
    test = cv2.imread('../results/scan-02.jpg')
    setCallback(test)