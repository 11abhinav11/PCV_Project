import numpy as np
import cv2
import math


def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]


def process_lines(image_src):
    img = cv2.imread(image_src)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    thresh1 = cv2.bitwise_not(thresh1)
    edges = cv2.Canny(thresh1, threshold1=50, threshold2=200, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=30)

    # l[0] - line; l[1] - angle
    for line in get_lines(lines):
        leftx, boty, rightx, topy = line
        cv2.line(img, (leftx, boty), (rightx, topy), (0, 0, 255), 6)

    # merge lines

    # ------------------
    # prepare
    _lines = []
    for l in get_lines(lines):
        _lines.append([(l[0], l[1]), (l[2], l[3])])

    # sort
    _lines_x = []
    _lines_y = []
    _lines_xy = []
    _lines_yx = []
    for line_i in _lines:
        orientation_i = math.atan2(
            (line_i[0][1]-line_i[1][1]), (line_i[0][0]-line_i[1][0]))

        if (abs(math.degrees(orientation_i)) >= 67.5) and abs(math.degrees(orientation_i)) < 112.5:
            _lines_y.append(line_i)
        elif (abs(math.degrees(orientation_i)) > 22.5 and abs(math.degrees(orientation_i)) < 67.5 and orientation_i < 0):
            _lines_xy.append(line_i)
        elif (abs(math.degrees(orientation_i)) >= 112.5 and abs(math.degrees(orientation_i)) <= 157.5 and orientation_i > 0):
            _lines_xy.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
    _lines_xy = sorted(_lines_xy, key=lambda _line: _line[0][1])
    _lines_yx = sorted(_lines_yx, key=lambda _line: _line[0][0])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)
    merged_lines_yx = merge_lines_pipeline_2(_lines_yx)
    merged_lines_xy = merge_lines_pipeline_2(_lines_xy)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)
    merged_lines_all.extend(merged_lines_yx)
    merged_lines_all.extend(merged_lines_xy)
    print("process groups lines", len(_lines), len(merged_lines_all))
    img_merged_lines = cv2.imread(image_src)
    img_merged_lines = cv2.resize(img_merged_lines, (0, 0), fx=0.5, fy=0.5)
    for line in merged_lines_all:
        cv2.line(img_merged_lines, (line[0][0], line[0][1]),
                 (line[1][0], line[1][1]), (0, 0, 255), 6)
    cv2.imshow("lines", img_merged_lines)
    cv2.waitKey(0)
    return merged_lines_all


def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2(
                        (line[0][1]-line[1][1]), (line[0][0]-line[1][0]))
                    orientation_j = math.atan2(
                        (line2[0][1]-line2[1][1]), (line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2(
                        (line[0][1]-line[1][1]), (line[0][0]-line[1][0]))
                    orientation_j = math.atan2(
                        (line2[0][1]-line2[1][1]), (line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        # lines[idx] = False
            # append new group
            super_lines.append(new_group)

    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final


def merge_lines_segments1(lines, use_log=False):
    if (len(lines) == 1):
        return lines[0]

    line_i = lines[0]

    # orientation
    orientation_i = math.atan2(
        (line_i[0][1]-line_i[1][1]), (line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):

        # sort by y
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:

        # sort by x
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points)-1]]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw


def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

    if (min(dist1, dist2, dist3, dist4) < 100):
        return True
    else:
        return False


def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude

# Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/


def DistancePointLine(px, py, x1, y1, x2, y2):
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    return min(dist1, dist2, dist3, dist4)


def pixel_radius(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = gray.shape
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=200)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            if a < width/3 and b < height/3:
                return r


def length_of_each_edge(ratio, lines):
    length = []
    for line in lines:
        pixel_length = math.sqrt(
            pow(line[0][0]-line[1][0], 2)+pow(line[0][1]-line[1][1], 2))
        length.append(pixel_length*ratio)
    return length


paths = ["D:/program files/q.png", "D:/program files/qq.jpg",
         "D:/program files/111.jpg", "D:/program files/222.jpg"]
for file in paths:
    lines = process_lines(file)
    print(lines)
    real_coin_size = 1.25

    pixel_coin_size = pixel_radius(file)
    print(pixel_coin_size)
    print(real_coin_size/pixel_coin_size)
    length = length_of_each_edge(real_coin_size/pixel_coin_size, lines)
    count = 0
    img = cv2.imread(file)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    for line in lines:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int((line[0][0]+line[1][0])/2), int((line[0][1]+line[1][1])/2))
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1
        img = cv2.putText(img, str(round(length[count], 2))+"cm", org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        count += 1

# Displaying the image
    cv2.imshow("output", img)
    cv2.waitKey(0)
