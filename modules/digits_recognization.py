import numpy as np
from .sheet_process import Sheet
import os
import pickle
import cv2 as cv
# from sklearn.decomposition import PCA
from scipy.ndimage import interpolation
from sklearn.cluster import KMeans


def recognize_digits_allsheets(
        paths_sheets,
        ROI,
        path_trained_models='trained_models',
        n_digits=6,
        path_result='results_coversheet/'):
    if not os.path.exists(path_result):
        os.mkdir(path_result)
    # SVM classifier
    clf = pickle.load(
        open(os.path.join(path_trained_models, 'SVCtrained.dms'), 'rb'))
    # PCA model
    pca = pickle.load(
        open(os.path.join(path_trained_models, "PCAtrained.dms"), 'rb'))
    print('Recognizing digit string on Cover sheets...')
    ids_student = []
    for i, path_sheet in enumerate(paths_sheets):
        sheet = Sheet(path_sheet)
        orginal_img = sheet.img_original.copy()
        img_digit_string = sheet.img_bi[ROI[1]:ROI[1] +
                                        ROI[3], ROI[0]:ROI[0]+ROI[2]].astype(np.uint8)*255
        img_digit_string = remove_spots(img_digit_string)
        imgs_digits, retvals = find_digits(img_digit_string, nb_digit=None)
        predicted_values = []
        str_predicted_values = []

        for idx, digit_img in enumerate(imgs_digits):
            digit_img = remove_spots(digit_img, min_size=None)
            digit_img = resize_digit(digit_img)
            digit_img = deskew(digit_img, rotate=False) > 10
            predicted_value = int(clf.predict(
                pca.transform(digit_img.reshape(1, -1))))
            predicted_values.append(predicted_value)
            str_predicted_values.append(str(predicted_value))
            x = ROI[0]+retvals[idx]['x']
            y = ROI[1]+retvals[idx]['y']
            w = retvals[idx]['w']
            h = retvals[idx]['h']
            color = np.random.randint(0, 255, 3).tolist()
            cv.rectangle(orginal_img, (x, y), (x+w, y+h), color)
            cv.putText(orginal_img, str(predicted_value), (x, y), cv.FONT_HERSHEY_PLAIN,
                       2, color, 2)
        id_string = ''.join(str_predicted_values)
        cv.imwrite(os.path.join(path_result, 'ID_' +
                                id_string+'.png'), orginal_img)
        ids_student.append(int(id_string))
        # print('{}th img, results:{}'.format(i, predicted_values))
    print('Finished! all processed cover sheets are saved unter the folder:{}'.format(
        path_result))
    return ids_student


def remove_spots(img, min_size=30):
    '''
    remove the spots in which the number of pixels are less than min_size
    @ param min_size int or None minimum size of the project
        if it is defined as none then only the object has the maximal number of pixels will be reserved
    '''
    digit_imgs = []
    img_copy = img.copy()
    nb_components, labels_mask, stats, _ = cv.connectedComponentsWithStats(
        img, connectivity=8)
    # iterate over each label but skip the background
    if min_size is not None:
        for label in range(1, nb_components):
            # check the number of the pixels of the object
            if stats[label, -1] < min_size:
                img_copy[labels_mask == label] = 0
    else:
        label_max = np.argmax(stats[1:, -1])+1
        img_copy = (labels_mask == label_max).astype(np.uint8)*255
    return img_copy


def moments(image):
    # A trick in numPy to create a mesh grid
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0*image)/totalImage  # mu_x
    m1 = np.sum(c1*image)/totalImage  # mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage  # var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage  # var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage  # covariance(x,y)
    # Notice that these are \mu_x, \mu_y respectively
    mu_vector = np.array([m0, m1])
    # Do you see a similarity between the covariance matrix
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix


def deskew(image, rotate=True):
    c, v = moments(image)
    if rotate:
        alpha = v[0, 1]/v[0, 0]
    else:
        alpha = 0
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


def resize_digit(img, output_size=28, zero_padding=3):
    img = remove_spots(img, min_size=None)
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, 'digit is broken'
    x, y, w, h = cv.boundingRect(contours[0])
    # extract the object
    img = img[y:y+h, x:x+w]
    h, w = img.shape
    ratio = h/w
    effective_size = output_size-zero_padding*2
    # resize the image and reserve the ratio of the object
    if ratio > 1:
        img = cv.resize(
            img,
            (int(np.round(effective_size/ratio)), effective_size))

    else:
        img = cv.resize(
            img,
            (effective_size, int(np.round(effective_size*ratio))))
    # zero padding
    output_img = np.zeros((output_size, output_size))
    left_upper = (output_size//2 -
                  img.shape[0]//2, output_size//2-img.shape[1]//2)
    output_img[left_upper[0]:left_upper[0]+img.shape[0],
               left_upper[1]:left_upper[1]+img.shape[1]] = img
    return output_img


def find_digits(img, nb_digit=6):
    outputs = []
    retvals = []
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        outputs.append(img[y:y+h, x:x+w].copy())
        retvals.append(
            {'x': x, 'y': y, 'w': w, 'h': h})
    # adjacent digits string cause the number of detected digits smaller than defined number
    if nb_digit is not None and len(outputs) < nb_digit:
        try:
            ws = [retval['w'] for retval in retvals]
            kmeans = KMeans(n_clusters=2).fit(np.array(ws).reshape(-1, 1))
            labels = kmeans.labels_
            pos_where = np.where(labels == 1)
            neg_where = np.where(labels == 0)
            # make sure the positive labeled objects are  wider
            if retvals[pos_where[0][0]]['w'] < retvals[neg_where[0][0]]['w']:
                tmp = neg_where
                neg_where = pos_where
                pos_where = tmp
    #         iterate to split adjacent digits
            for idx in pos_where[0]:
                # relative position to splite the image
                w = outputs[idx].shape[0]
                x_split = np.argmin(
                    outputs[idx][:, int(w/4):int(w*3/4)].sum(axis=0))+int(w/4)
                digit1 = outputs[idx][:, :x_split]
                digit2 = outputs[idx][:, x_split:]
                outputs.insert(idx+1, digit1)
                retvals.insert(
                    idx+1,
                    {'x': retvals[idx]['x'], 'y': retvals[idx]['w'], 'w': x_split, 'h': retvals[idx]['h']})
                outputs.insert(idx+2, digit2)
                retvals.insert(
                    idx+2,
                    {'x': retvals[idx]['x']+x_split, 'y': retvals[idx]['w'], 'w': w-x_split, 'h': retvals[idx]['h']})
                del outputs[idx]
                del retvals[idx]
        except:
            pass
    zipped_outputs = zip(outputs, retvals)
    sorted_zip = sorted(zipped_outputs, key=lambda t: t[1]['x'])
    outputs = [item[0] for item in sorted_zip]
    retvals = [item[1]for item in sorted_zip]
    return outputs, retvals
