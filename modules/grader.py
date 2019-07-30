import numpy as np
import pandas as pd
import cv2 as cv
import os
from .sheet_process import AnswerSheet
from .Interactive import setCallback


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
        os.path.join(path_imgs_save, 'ID_'+str(id_student)+'.png'), img_save)
    return points_sum


def grade_sheets(paths_images,
                 ids_student,
                 solutions,
                 p_solutions,
                 semi_mode_on=True,
                 path_imgs_save='log_imgs/'
                 ):
    # get ids_student, solutions and points of solutions
    points_students = []
    for id_student, path_image in zip(ids_student, paths_images):
        answer_sheet = AnswerSheet(path_image)
        answer_sheet.run()
        # map = answer_sheet.default_map
        answers_student = answer_sheet.answers.copy()
        answer_sheet_to_edit = answer_sheet.img_cross_detected.copy()
        # print(answer_sheet.estimate_chopped_lines_center_h())
        # semi-automatic mode
        if semi_mode_on:
            answers, map_result, img = setCallback(
                answer_sheet_to_edit,
                answer_sheet.table,
                answer_sheet.img_original,
                answer_sheet.estimate_chopped_lines_center_h(),
                answer_sheet.default_map,
                answers_student)
        # full-automatic mode
        else:
            answers = answers_student
            map_result = None
            img = answer_sheet_to_edit

        # print(answers.sum())

        coordinates = [None]*len(solutions)
        if map_result is not None:
            for row in map_result:
                answers_student[row[0]-1,
                                :] = answers[row[1]-1, :]
                coordinates[row[0]-1] = [answer_sheet.table[row[1]][4, -1, :, 0] +
                                         answer_sheet.table_info['cell_w'],
                                         answer_sheet.table[row[1]][4, -1, :, 1] +
                                         int(answer_sheet.table_info['cell_h']/2)]
        for idx in range(len(solutions)):
            # TOBEDELETED
            if answer_sheet.table[idx+1] is not None:
                coordinates[idx] = [answer_sheet.table[idx+1][4, -1, :, 0] +
                                    answer_sheet.table_info['cell_w'],
                                    answer_sheet.table[idx+1][4, -1, :, 1] +
                                    int(answer_sheet.table_info['cell_h']/2)]

        points = grad(id_student,
                      answers_student,
                      solutions,
                      path_imgs_save,
                      img,
                      coordinates,
                      p_solutions
                      )
        points_students.append(points)
        # save the results
    # idsANDscores = np.array([ids_student, points_students]).T
    df = pd.DataFrame(
        {'Student ID': ids_student,
         'Scores': points_students})
    df.to_csv(
        os.path.join(os.path.dirname(path_imgs_save),
                     'students_IDS_scores.csv'),
        index=False)

    # np.savetxt(
    #     os.path.join(os.path.dirname(path_imgs_save),
    #                  'students_IDS_scores.csv'),
    #     idsANDscores,
    #     fmt='%.1f',
    #     delimiter='\t\t',
    #     header='Student ID \t score'
    # )
