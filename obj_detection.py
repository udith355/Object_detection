import cv2 as cv
import imutils
from imutils import paths
import numpy as np
import numpy as np

#
# input sample video file name
#

# file_name = 'vtest.avi'
# tikak hari
# file_name='car_race.mp4'
# hari
# file_name='test6.mp4'
file_name = 'test5.mp4'

#
# Press Q to close the window
#


# Threshold values
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_SIMPLEX

# The height of the vehical
focal_length = 0.477
real_car_height = 1.6
real_truck_height = 3.5
real_bus_height = 3.2
real_motorbike_height = 1.2

# importing object names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


#  setttng up opencv net
def setUpOpenCVNet():
    yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(yoloNet)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
    return yoloNet, model


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    edged = cv.Canny(gray, 35, 125)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    # compute the bounding box of the of the paper region and return it
    return cv.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


# Detect objects and show identified object
def ObjectDetector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s" % (class_names[classid])

        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 10), fonts, 0.5, color, 2)
    # ===========================================================================================
    marker = find_marker(image)
    KNOWN_DISTANCE = 16.0
    KNOWN_WIDTH = 6.0
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    # =======================================================================================================
    for i, b in enumerate(boxes):

        if classes[i] == 2 or classes[i] == 7 or classes[i] == 8:
            if scores[i] >= 0.6:
                apx_distance = distance_to_camera(KNOWN_WIDTH, focalLength, abs(boxes[i][1] - boxes[i][3]))
                mid_x = (boxes[i][1] + boxes[i][3]) / 2

                mid_y = (boxes[i][0] + boxes[i][2]) / 2

                apx_distance = round(((1 - (boxes[i][3] - boxes[i][1]))), 1)

            if (apx_distance) <= 350:
                if (mid_x) > (100) and (mid_x) < (400):
                    cv.putText(image, 'WARNING!!!'.format(apx_distance=apx_distance),
                               (box[0] + 10, box[1] + 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv.rectangle(image, box, (0, 0, 255), 2)




# capture and call needed functions
def start_processing():
    camera = cv.VideoCapture('sample_videos/' + file_name)
    counter = 0
    capture = False
    # number = 0
    while True:
        ret, frame = camera.read()

        orignal = frame.copy()

        ObjectDetector(frame)


        cv.imshow('frame', frame)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
    cv.destroyAllWindows()


yoloNet, model = setUpOpenCVNet();
start_processing()
