import os
import numpy as np


def random_generate_ids(dir_sheets):
    names_image = sorted(os.listdir(dir_sheets))
    n_ids = int(len(names_image)/2)
    ids = np.random.randint(300000, 400000, size=(n_ids, 1))
    np.savetxt('student_ids.csv', ids)


if __name__ == '__main__':
    random_generate_ids('scan/')
