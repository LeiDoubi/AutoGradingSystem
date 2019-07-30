import numpy as np


class DigitsString:
    def __init__(self, img):
        assert isinstance(img, np.ndarray),\
            'Initialization failed, the type of the param content is %s,\
             not a numpy array' % type(img)
        self.img = img
        self.digits = None

    def split_digits(self):
        digits_info = {
            'img': None,
            'n_digits': 1}


def predict_single_digit(img):
    pass
