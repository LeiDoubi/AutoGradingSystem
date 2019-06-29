import numpy as np
import cv2 as cv
import os


def grad(id_student,
         answers_student,
         answers_stand,
         path_save,
         img_cross_detected,
         coordinates,
         points_single_answer=2.0
         ):
    img_save = img_cross_detected.copy()
    points_sum = 0.0
    for i, row in enumerate(answers_student):
        if row == answers_stand[i]:
            points_current = row.sum()*points_single_answer
        else:
            correct_answers = np.logical_and(row, answers_stand)
            wrong_answers = row - correct_answers
            points_current = np.max(
                correct_answers.sum()*points_single_answer
                - wrong_answers.sum()*points_single_answer/2.0, 0)
        cv.putText(
            img_save,
            str(points_current),
            (coordinates[i, 0],
             coordinates[i, 1]),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (20, 20, 255),
            4)
        points_sum += points_current
    cv.putText(
        img_save,
        str(points_sum),
        (coordinates[-1, 0],
         2*coordinates[-1, 1]-coordinates[-2, 1]),
        cv.FONT_HERSHEY_PLAIN,
        3,
        (20, 20, 255),
        4)
    cv.imwrite(
        os.path.join(path_save, str(id_student)+'.png'), img_save)
    return points_sum
