# import time
import numpy as np
import cv2 as cv
import scipy.spatial.distance as scid


def detectCrossinCell(img, rect):
    '''
    detectCrossinCell(img, rect)=> iscrossincell,isabnormal, lines
    @param iscrossincell : bool
    @param isabnormal  : bool
    @param lines all found lines: ndarry, shape: (N,1,4)
    @param img: binary image , it shoud have the type of numpy.ndarry \n
    @param rect: shape(4,1,2), the coordinates of the corners of the rectangle

    '''
    iscrossincell = False
    isabnormal = False
    lines = None
    if not isinstance(img, np.ndarray) or not isinstance(rect, np.ndarray):
        raise Exception(' the type of image or rectangular is not np.ndarray')
    if rect.shape[0] != 4:
        raise Exception(
            'The given rect have the information of {} corners'
            .format(rect.shape[0]))
    # find the left-upper corner and right-lower corner
    rect_float = rect.astype('float64')
    index_sorted = np.argsort(
        (rect_float[:, :, 0]**2+rect_float[:, :, 1]**2).flatten())
    leftupperCorn = rect[index_sorted[0], :, :]
    rightlowerCorn = rect[index_sorted[-1], :, :]
    erode = 7
    lines = cv.HoughLinesP(
        img[leftupperCorn[0, 1]+erode:rightlowerCorn[0, 1]-erode,
            leftupperCorn[0, 0]+erode:rightlowerCorn[0, 0]-erode],
        1, np.pi/20, 7, minLineLength=8)
    if lines is None:
        return iscrossincell, isabnormal, lines
    else:
        # add offset to get absolute coordinates
        lines[:, :, [0, 2]] = leftupperCorn[0, 0]+lines[:, :, [0, 2]]
        lines[:, :, [1, 3]] = leftupperCorn[0, 1]+lines[:, :, [1, 3]]
        segmented_lines = _segmentLines(lines)
        #  segmented_lines is None means that the
        #  lines can not be segmented into 2 groups
        if segmented_lines is None:
            isabnormal = True
            return iscrossincell, isabnormal, lines
        else:
            intersections = _findIntersections2LineGroup(
                segmented_lines[0], segmented_lines[1])
            # _checkOneCluster(intersections)
            return True, isabnormal, lines


def _checkOneCluster(X):
    '''
    '''
    distance_matrix = scid.cdist(X, X)
    pass


def _segmentLines(lines):
    '''
    _segment_lines(lines)=> lines_pos,lines_neg or None \n
    if the lines can not be grouped into 2 groups
    segment the lines according to the slopes
    return  lines with positive slope, lines with negative slope
    '''
    slopes = (lines[:, :, 3]-lines[:, :, 1])/(lines[:, :, 2]-lines[:, :, 0])
    slope_positive = slopes > 0
    lines_pos, lines_neg = lines[slope_positive, :], lines[~slope_positive, :]
    if lines_pos.shape[0] != 0 and lines_neg.shape[0] != 0:
        return lines_pos, lines_neg
    else:
        return None


def _findIntersections2LineGroup(line_groupA, line_groupB):
    '''
    _intersection_2LineGroup(line_groupA,line_groupB)=> ndarry shape:(N,2)
    return all intersections of lines in two groups
    '''
    intersections = []
    for line_a in line_groupA:
        for line_b in line_groupB:
            point = _intersection(line_a, line_b)
            if point is not None:
                intersections.append(point)
    return np.array(intersections)


def _intersection(lineA, lineB, isLineSegment=True):
    '''
    calculate the intersction of finite lines
    lineA (x1,y1,x2,y2)
    lineB (x3,y3,x4,y4)
    The this problem can be solved by :
    --               --  --   --      --                   --
    |   y2-y1   x1-x2 |  |  x  |    = | (y2-y1)x1+(x1-x2)y1 |
    |   y4-y3   x3-x4 |  |  y  |      | (y4-y3)x3+(x3-x4)y3 |
    --               --  --   --      --                   --
    return (x, y) if intersction exists else None
    '''
    x1, y1, x2, y2 = lineA
    x3, y3, x4, y4 = lineB
    coff = np.array([[y2-y1, x1-x2], [y4-y3, x3-x4]])
    b = np. array([(y2-y1)*x1+(x1-x2)*y1, (y4-y3)*x3+(x3-x4)*y3])
    try:
        point = np.linalg.solve(coff, b)
    except np.linalg.LinAlgError:
        return None
    if not isLineSegment:
        return point
    else:
        # make sure the intersection is on the both line segments
        sorted_x = np.sort([[x1, x2], [x3, x4]])
        if point[0] <= sorted_x[0, 1]\
                and point[0] >= sorted_x[0, 0]\
                and point[0] <= sorted_x[1, 1]\
                and point[0] >= sorted_x[1, 0]:
            return point
        else:
            return None


if __name__ == '__main__':
    print(intersection((1, 1, 10, 10), (2, 0, 2, 1)))
