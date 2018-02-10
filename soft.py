import numpy as np
from scipy import ndimage
import math
import cv2
import vector_fundza
from convNN import createModel
import random
import sys

class Element:
    def __init__(self):
        self.center = (0,0)
        self. history = []
        self.value = 0
        self.bluePassed = False
        self.greenPassed = False
        self.frameNo = 0

class History:
    def __init__(self):
        self.bluePassed = False
        self.greenPassed = False
        self.center = (0, 0)
        self.frameNo = 0

def find_lines(image) :

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([60, 100, 100])
    upper_green = np.array([70, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    res_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    res_green = cv2.bitwise_and(image, image, mask=mask_green)
    #cv2.imshow("res bl", res_blue)
    cany_blue = cv2.Canny(res_blue, 50, 200, None, 3)
    cany_green = cv2.Canny(res_green, 50, 200, None, 3)
    #cv2.imshow("cany bl", cany_blue)
    lines_blue = cv2.HoughLinesP(cany_blue, 1, np.pi / 180, 50, None, 50, 20)
    lines_green = cv2.HoughLinesP(cany_green, 1, np.pi / 180, 50, None, 50, 20)
    line_points = [[(0,0), (0,0)],
                           [(0,0), (0,0)]]

    distance_max = 0
    if lines_blue is not None:
        for i in range(0, len(lines_blue)):
            line = lines_blue[i][0]
            distance = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if distance > distance_max:
                line_points[0][0] = (line[0], line[1])
                line_points[0][1] = (line[2], line[3])
                distance_max = distance
            cv2.line(image, (line_points[0][0][0], line_points[0][0][1]), (line_points[0][1][0], line_points[0][1][1]),
                   (0, 0, 255), 1, cv2.LINE_AA)
    distance_max = 0
    if lines_green is not None:
        for i in range(0, len(lines_green)):
            line = lines_green[i][0]
            distance = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if distance > distance_max:
                line_points[1][0] = (line[0], line[1])
                line_points[1][1] = (line[2], line[3])
                distance_max = distance
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv2.LINE_AA)
    return line_points

def train_network() :
    image = cv2.imread('digits.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.pyrDown(image)
    #cv2.imshow('Digits', small)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)
    #print("Shape of cell arrays" + str(x.shape))

    train = x[:,:70].reshape(-1, 400).astype(np.float32)
    test = x[:,70:100].reshape(-1, 400).astype(np.float32)

    cifre = [0,1,2,3,4,5,6,7,8,9]

    train_lbl = np.repeat(cifre, 350)[:,np.newaxis]
    test_lbl = np.repeat(cifre, 150)[:,np.newaxis]

    #print("train")
    #print(train)
    #print("train lbl")
    #print(train_lbl)

    knn = cv2.ml.KNearest_create()
    knn.train(train,cv2.ml.ROW_SAMPLE, train_lbl)

    ret, result, neighbours, distance = knn.findNearest(test, k=3)

    matches = result == test_lbl
    correct = np.count_nonzero(matches)
    accuracy = correct * (100.0 / result.size)
    print("accuracy is = %.2f" % accuracy)
    return knn

def find_center(contur):
    # M = cv2.moments(contur)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # return (cx, cy)
    (x, y ,w ,h) = contur
    cx = int(x+ w/2)
    cy = int(y+ h/2)
    return int(cx),int(cy)

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed

    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        # print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = (height - width) / 2
            # print("Padding = ", pad)
            pad = int(pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width - height) / 2
            #print("Padding = ", pad)
            pad = int(pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0,cv2.BORDER_CONSTANT, value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square

def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions

    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg

def intersecting_blue(image, line_points, center):
    #(x, y, w, h) = contures
    blue_line_mask = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    conture_mask = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    cv2.circle(conture_mask, center, 3, (0, 0, 255), 2)

    cv2.line(blue_line_mask, (line_points[0][0][0], line_points[0][0][1]),
             (line_points[0][1][0], line_points[0][1][1]), (0, 0, 255), 1, cv2.LINE_AA)
    #cv2.imshow('line', blue_line_mask)
    #cv2.line(image, (line_points[0][0][0], line_points[0][0][1]), (line_points[0][1][0], line_points[0][1][1]),
            # (0, 0, 255), 1, cv2.LINE_AA)
    intersect =  1 in np.logical_and(conture_mask, blue_line_mask)

    return intersect

def intersecting_green(image, line_points,  center):
    #(x, y, w, h) = contures
    green_line_mask = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    conture_mask = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    cv2.circle(conture_mask, center, 3, (0, 0, 255), 2)

    cv2.line(green_line_mask, (line_points[1][0][0], line_points[1][0][1]),
             (line_points[1][1][0], line_points[1][1][1]), (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.imshow('line', blue_line_mask)
    # cv2.line(image, (line_points[0][0][0], line_points[0][0][1]), (line_points[0][1][0], line_points[0][1][1]),
    # (0, 0, 255), 1, cv2.LINE_AA)
    intersect = 1 in np.logical_and(conture_mask, green_line_mask)
    return intersect

def deskew(img):
    m = cv2.moments(img)
    SZ = 28
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def findElement(elements, element):
    indexes = []
    i = 0
    for el in elements:
        (eX,eY) = element.center
        (hX,hY) = el.center
        distance = math.sqrt(math.pow((eX - hX),2) + math.pow((eY - hY),2))
        if distance < 20:
            indexes.append(i)
        i += 1
    return indexes
def crop(number):
    ret, thresh = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return number
    contourIndex = np.argmax(areas)
    [x, y, w, h] = cv2.boundingRect(contours[contourIndex])
    cropped = number[y:y + h + 1, x:x + w + 1]
    cropped = cv2.resize(cropped, (28,28), interpolation=cv2.INTER_AREA)
    return cropped
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
def deleteBlanks(gray):
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 28.0 / rows
        rows = 28
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 28.0 / cols
        cols = 28
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    return gray

def detectNumbers(image):
    lower = np.array([230, 230, 230], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    image = cv2.bitwise_and(image,image,mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.GaussianBlur(gray, (5, 5), 0)

    #edged = cv2.Canny(blurred, 10, 200)
    #ret2, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY)
    #ret,edged = cv2.threshold(blurred,90,255, cv2.THRESH)
    # Find Contours
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours_ret = []
    for i,c in enumerate(contours):
        # compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 20 and area < 1200 and h > 10:
            coordinates = (x, y, w, h)
            contours_ret.append(coordinates)
    #cv2.imshow("thresh", thresh)
    #contours_ret = sorted(contours_ret, key=lambda tup: tup[1], reverse=True)
    return contours_ret
def predictNumber(image,center,knn,contour=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("predict", gray)
    if contour != None:
        (x,y,w,h) = contour
        roi = image[y:y + h + 1, x:x + w + 1]
    else:
        roi = gray[center[1]-12:center[1]+12, center[0]-12:center[0]+12]
    (thresh, roi) = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #roi = deskew(roi)
    cropped = crop(roi)
    cropped = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("blanks deleted" + str(random.randint(0,1000)), cropped)
    # if cv2.waitKey(1) == 13:
    #     return
    # cv2.imshow("predict", gray)
    # if cv2.waitKey(1) == 13:
    #     return
    #shiftx, shifty = getBestShift(roi)
    #final = shift(roi, shiftx, shifty)
    #cv2.imshow("shifted", final)
    #dimData = np.prod(roi.shape[1:])
    #roiOneRow = roi.reshape(1, dimData*dimData).astype('float32')
    cropped = cropped / 255
    #final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    #roi_hog_fd = hog(final.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    #nbr = knn.predict(np.array([roi_hog_fd], 'float64'))
    number = knn.predict_classes(cropped.reshape(1,28,28,1))
    return int(number)

if __name__ == '__main__':
    videoName = 'video-' + str(sys.argv[1])
    cap = cv2.VideoCapture(videoName + '.avi')
    kernel = np.ones((3,3),np.uint8)
    success, frame = cap.read()

    classifier = createModel((28,28,1),10)
    classifier.load_weights('cnnKerasWeights.h5')
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    line_points = find_lines(opening)

    numbers = []
    frameNo = 0
    sum = 0
    elements = []
    while success:
        success, frame = cap.read()
        if not success:
            break

        #opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        #gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
        if frameNo % 3 == 0:
            contours = detectNumbers(frame)
        for index, contour in enumerate(contours):
            element = Element()
            element.center = find_center(contour)
            element.frameNo = frameNo
            indexes = findElement(elements, element)
            if len(indexes) == 0:
                element.value = predictNumber(frame, element.center, knn=classifier)
                elements.append(element)
            elif len(indexes) == 1:
                #update elementa i dodavanje istorije
                h = History()
                h.frameNo = frameNo
                h.center = element.center
                elements[indexes[0]].history.append(h)
                elements[indexes[0]].frameNo = frameNo
                elements[indexes[0]].center = element.center

        for element in elements:
            if (frameNo - element.frameNo) > 10:
                continue
            greenPassedBefore = element.greenPassed
            bluePassedBefore = element.bluePassed
            color = [0, 0, 255]
            if not bluePassedBefore:
                distBl, pnt, rBl = vector_fundza.pnt2line(element.center, line_points[0][0], line_points[0][1])
                cv2.line(frame, (line_points[0][0][0], line_points[0][0][1]),
                         (line_points[0][1][0], line_points[0][1][1]), (0, 0, 255), 1, cv2.LINE_AA)
            if not greenPassedBefore:
                distGr, pnt, rGr = vector_fundza.pnt2line(element.center, line_points[1][0], line_points[1][1])
                cv2.line(frame, (line_points[1][0][0], line_points[1][0][1]),
                         (line_points[1][1][0], line_points[1][1][1]), (0, 0, 255), 1, cv2.LINE_AA)
            if distBl < 12.0 and not bluePassedBefore and rBl == 1:
                sum += int(element.value)
                element.bluePassed = True
                color = [220,0,0]
            if distGr < 12.0 and not greenPassedBefore and rGr == 1:
                sum -= int(element.value)
                element.greenPassed = True
                color = [0, 220, 0]
            cv2.circle(frame, element.center, 15, color, 2)
            cv2.putText(frame, str(element.value), (element.center[0] + 12, element.center[1] + 12),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            for hist in element.history:
                t = frameNo - hist.frameNo
                if (t < 200):
                    cv2.circle(frame, hist.center, 1, (255, 255, 255), 1)

        cv2.putText(frame, "Sum: " + str(sum) + "Frame number:" + str(frameNo), (15, 20), cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 13:
            break
        frameNo += 1
        #time.sleep(0.05)

    cv2.destroyAllWindows()
    cap.release()
    lines = []
    with open("out.txt","r") as file:
        data = file.read()
        lines = data.split('\n')
    f = open('out.txt', 'w')
    append = True
    if len(lines) < 2:
        lines = []
        lines.append('RA 172/2014 Milorad Trninic')
        lines.append('file sum')
    for i,line in enumerate(lines):
        if i == 0:
            continue
        if videoName in line:
            lines[i] = '\n'+videoName + '.avi\t' + str(sum)
            append = False
        else:
            lines[i] = '\n' + lines[i]
    if append:
        lines.append('\n'+videoName + '.avi\t' + str(sum))
    f.writelines(lines)
    f.close()
    # f = open('out.txt', 'a')
    # f.write('\n'+videoName + '.avi\t' + str(sum))
    # f.close()