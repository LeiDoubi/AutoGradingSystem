import numpy as np
import cv2 as cv
import os


def grad(id_student,
         answers_student,
         answers_stand,
         path_imgs_save,
         img_cross_detected,
         coordinates,
         points_answers=2
         ):
    if not isinstance(points_answers, (int, float, np.ndarray)):
        raise TypeError('type of points_answers is none of \
                         int,float and numpy array ')
    elif not isinstance(points_answers, np.ndarray):
        points_answers = np.array([points_answers]*len(id_student))
    if not os.path.exists(path_imgs_save):
        os.mkdir(path_imgs_save)
    img_save = img_cross_detected.copy()
    points_sum = 0.0
    for i, row in enumerate(answers_stand):
        if np.array_equal(row, answers_student[i]):
            points_current = row.sum()*points_answers[i]
        else:
            correct_answers = np.logical_and(row, answers_student[i])
            wrong_answers = answers_student[i] ^ correct_answers
            points_current = np.max(
                [correct_answers.sum()*points_answers[i]
                 - wrong_answers.sum()*points_answers[i]/2, 0])
        if points_current < 0:
            pass
            print('test')
        if coordinates[i] is not None:
            cv.putText(
                img_save,
                str(points_current),
                (coordinates[i][0],
                 coordinates[i][1]),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (20, 20, 255),
                4)
            points_sum += points_current
    for index in reversed(range(len(coordinates))):
        if coordinates[index] is not None:
            cv.putText(
                img_save,
                'Sum: '+str(points_sum),
                (coordinates[index][0],
                 coordinates[index][1]+100),
                cv.FONT_HERSHEY_PLAIN,
                5,
                (100, 100, 255),
                5)
            cv.putText(
                img_save,
                'ID: '+str(id_student),
                (coordinates[index][0],
                 coordinates[index][1]+200),
                cv.FONT_HERSHEY_PLAIN,
                5,
                (100, 100, 255),
                5)
            break
    cv.imwrite(
        os.path.join(path_imgs_save, str(id_student)+'.png'), img_save)
    return points_sum
