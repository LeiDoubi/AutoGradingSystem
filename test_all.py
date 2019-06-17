import glob
import os
from modules.sheet_process import *

imgs_dir_path = 'scan/'
os.mkdir('results')
imgs_paths = glob.glob(imgs_dir_path+'*.jpg')
for index in range(1, len(imgs_paths), 2):
    answer_sheet = AnswerSheet(imgs_paths[index])
    answer_sheet.run()