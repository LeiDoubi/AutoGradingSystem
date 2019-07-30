from modules.grader import grade_sheets
import numpy as np
import pandas as pd
from modules.Interactive import selectROI
from modules.digits_recognization import recognize_digits_allsheets
import os
import cv2 as cv
import shutil


def read_student_ids(path_ids):
    return np.loadtxt(path_ids, dtype=np.int32)


def get_solutions_points(path_solution):
    sheet = pd.read_excel(path_solution)
    solutions = sheet.iloc[:, 1:-1].fillna(0).to_numpy()
    # covert solutions to boolean numpy array in which check mark
    # represented with True
    solutions = solutions != 0
    points_solutions = sheet.iloc[:, -1].to_numpy(dtype=np.int8)
    return solutions, points_solutions


if __name__ == '__main__':
    # if set it to True, then for each answer sheet human intervention are required
    semi_mode_on = False
    #
    digit_recognize_on = True
    # directory for the outputs
    output_dir = 'all_results'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    # if the recognization of the digits string is not need then
    # the path of xlsx file that save students IDs should be given here
    if not digit_recognize_on:
        path_student_ids = 'inputs/student_ids_example.csv'

    path_solution = 'inputs/solution_example.xlsx'
    dir_CoverSheets = 'scans/'
    dir_AnswerSheets = 'scans/'
    # if all scans locate in same directory then split them
    if dir_AnswerSheets == dir_CoverSheets:
        names_Sheets = sorted(os.listdir(dir_CoverSheets))
        paths_Sheets = [os.path.join(dir_CoverSheets, name)
                        for name in names_Sheets]
        paths_AnswerSheets = paths_Sheets[1::2]
        paths_CoverSheets = paths_Sheets[::2]
    else:
        names_CoverSheets = sorted(os.listdir(dir_CoverSheets))
        paths_CoverSheets = [os.path.join(dir_CoverSheets, name)
                             for name in names_CoverSheets]
        names_AnswerSheets = sorted(os.listdir(dir_AnswerSheets))
        paths_AnswerSheets = [os.path.join(dir_AnswerSheets, name)
                              for name in names_AnswerSheets]
    # save the points of each question as a list
    points_solutions = get_solutions_points(path_solution)
    # do digit recognization on each cover sheet
    if digit_recognize_on:
        # mannually choose a the region of interest
        hint = '''Please select the region that contains the digits string by draging a rectangular!!
you can left click on the image to reselect the region and press ENTER to confirm your selection'''
        print(hint)
        ROI = selectROI(paths_CoverSheets[0])
        # apply ROI to each cover sheet
        ids_student = recognize_digits_allsheets(
            paths_CoverSheets,
            ROI,
            path_result=os.path.join(output_dir, 'CoverSheets'))
    # read the students ids from a xlsx file
    else:
        ids_student = read_student_ids(path_student_ids)
    solutions, points_solutions = get_solutions_points(path_solution)
    # grade the answer sheets
    grade_sheets(
        paths_AnswerSheets,
        ids_student,
        solutions,
        points_solutions,
        semi_mode_on,
        path_imgs_save=os.path.join(output_dir, 'AnswerSheets'))
