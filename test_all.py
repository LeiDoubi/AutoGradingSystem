import os
from modules.sheet_process import AnswerSheet

dir_image = 'scan/'
names_image = sorted(os.listdir(dir_image))
paths_image = [os.path.join(dir_image, name) for name in names_image]
for index in range(1, 2, 2):
    answer_sheet = AnswerSheet(paths_image[index])
    answer_sheet.run() 
    print(answer_sheet.answers)
    print(answer_sheet.default_map)