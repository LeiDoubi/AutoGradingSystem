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
    intersections = None
    if not isinstance(img, np.ndarray) or not isinstance(rect, np.ndarray):
        raise Exception(' the type of image or rectangular is not np.ndarray')
    if rect.shape[0] != 4:
        raise Exception(
            'The given rect have the information of {} corners'
            .format(rect.shape[0]))
    # find the left-upper corner and right-lower corner
    leftupperCorn = rect[0, ...]
    rightlowerCorn = rect[3, ...]
    erode = 2
    lines = cv.HoughLinesP(
        img[leftupperCorn[0, 1]+erode:rightlowerCorn[0, 1]-erode,
            leftupperCorn[0, 0]+erode:rightlowerCorn[0, 0]-erode],
        1, np.pi/150, 20, minLineLength=4, maxLineGap=3)
    if lines is None:
        return iscrossincell, isabnormal, lines, intersections
    else:
        # add offset to get absolute coordinates
        lines[:, :, [0, 2]] = leftupperCorn[0, 0]+erode+lines[:, :, [0, 2]]
        lines[:, :, [1, 3]] = leftupperCorn[0, 1]+erode+lines[:, :, [1, 3]]
        segmented_lines = _segmentLines(lines)
        #  segmented_lines is None means that the
        #  lines can not be segmented into 2 groups
        if segmented_lines is None:
            isabnormal = True
            return iscrossincell, isabnormal, lines, intersections
        else:
            intersections = _findIntersections2LineGroups(
                segmented_lines[0], segmented_lines[1])
            if intersections.shape[0] != 0:
                # too many intersections means that student
                # wanna to correct answer
                if intersections.shape[0] > 30:
                    iscrossincell = False
                else:
                    iscrossincell = isLineinOneCluster(intersections)
                    # in case all intersections though can be segmented
                    # into one cluster but is not concentrated
                    if iscrossincell:
                        center = np.mean(intersections, axis=0)
                        centered_intersections = intersections - center
                        iscrossincell = np.max(np.sum(
                            centered_intersections**2, axis=1
                        )) < 35
                return iscrossincell, isabnormal, lines, intersections
            else:
                intersections = _findIntersections2LineGroup2(
                    segmented_lines[0], segmented_lines[1])
                if intersections is not None:
                    iscrossincell = True
                    return iscrossincell, isabnormal, lines, intersections
                else:
                    return False, isabnormal, lines, intersections


def isLineinOneCluster(X):
    '''
    '''
    distance_threshold = 5
    distance_matrix = scid.cdist(X, X)
    result = {}
    current_stack = [0]
    while len(current_stack) != 0:
        index_currentpoint = current_stack.pop()
        result[index_currentpoint] = None
        index_nearpoints = np.argwhere(
            np.logical_and(
                distance_matrix[index_currentpoint] < distance_threshold,
                distance_matrix[index_currentpoint] != 0
            )
        )
        for index_nearpoint in index_nearpoints:
            int_index_nearpoint = int(index_nearpoint)
            if int_index_nearpoint not in result:
                result[int_index_nearpoint] = None
                current_stack.append(int_index_nearpoint)
    return len(result) == X.shape[0]


def _segmentLines(lines):
    '''
    _segment_lines(lines)=> lines_pos,lines_neg or None \n
    if the lines can not be grouped into 2 groups
    segment the lines according to the slopes
    return  lines with positive slope, lines with negative slope
    '''
    slopes = (lines[:, :, 3]-lines[:, :, 1])/(lines[:, :, 2]-lines[:, :, 0])
    slope_positive = slopes >= 0.1
    slope_negative = slopes < -0.1
    lines_pos, lines_neg = lines[slope_positive, :], lines[slope_negative, :]
    if lines_pos.shape[0] != 0 and lines_neg.shape[0] != 0:
        return lines_pos, lines_neg
    else:
        return None


def _findIntersections2LineGroups(line_groupA, line_groupB):
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
    allowed_error = 2
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
        if point[0] <= sorted_x[0, 1]+allowed_error\
                and point[0] >= sorted_x[0, 0]-allowed_error\
                and point[0] <= sorted_x[1, 1]+allowed_error\
                and point[0] >= sorted_x[1, 0]-allowed_error:
            return point
        else:
            return None


def _findIntersections2LineGroup2(line_groupA, line_groupB):
    points_pos = []
    points_neg = []

    for line_a in line_groupA:
        x1, y1, x2, y2 = line_a
        points_pos.append([x1, y1])
        points_pos.append([x2, y2])

    for line_b in line_groupB:
        x1, y1, x2, y2 = line_b
        points_neg.append([x1, y1])
        points_neg.append([x2, y2])
    points_neg_np = np.array(points_neg)
    points_pos_np = np.array(points_pos)

    # fit lines, vx, vy are normalize vector, x, y is a point on the line
    [vx_pos, vy_pos, x_pos, y_pos] = cv.fitLine(
        points_pos_np, cv.DIST_HUBER, 0, 0.01, 0.01)
    [vx_neg, vy_neg, x_neg, y_neg] = cv.fitLine(
        points_neg_np, cv.DIST_HUBER, 0, 0.01, 0.01)

    # y = k*x+b calculate k, b
    slope_pos = vy_pos/vx_pos
    slope_neg = vy_neg/vx_neg
    bias_pos = y_pos - slope_pos * x_pos
    bias_neg = y_neg - slope_neg * x_neg
    # intersection of these 2 lines
    point_x = (bias_neg-bias_pos)/(slope_pos-slope_neg)
    point_y = point_x * slope_pos + bias_pos
    # sort, in order to find out the 4 verticies of the cross
    points_pos.sort()
    points_neg.sort()
    bottomleft = points_pos[0]
    bottomright = points_neg[len(points_neg)-1]
    upperleft = points_neg[0]
    upperright = points_pos[len(points_pos)-1]

    # error set to 6 pixels, in order to avoid the case that hough transform
    # can only detect the lines on bottom(or others) half of the cross
    if point_x <= bottomright[0]+6\
            and point_x >= bottomleft[0]-6\
            and point_y <= upperleft[1]+6\
            and point_y >= bottomleft[1]-6:
        point = [int(point_x), int(point_y)]
        return point
    else:
        return None


if __name__ == '__main__':
    pass
