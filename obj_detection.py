import cv2 as cv
import imutils
from imutils import paths
import numpy as np

#
# input sample video file name
#


# file_name='test6.mp4'
file_name = 'test5.mp4'
# file_name = 'test4.mp4'

c = 0


#
# Press Q to close the window
#


# Threshold values
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# PINK = (147, 20, 255)
# ORANGE = (0, 69, 255)
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
# Color the background
def draw_text(img, text,
          font=cv.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=int(1.0),
          font_thickness=2,
          text_color=(255, 0, 0),
          text_color_bg=(0, 255, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size


# Detect objects and show identified object
def ObjectDetector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s" % (class_names[classid])

        cv.rectangle(image, box, color, 2)
        draw_text(image, label, cv.FONT_HERSHEY_PLAIN, (box[0], box[1] - 20), 2, 2)

        # cv.putText(image, label, (box[0], box[1] - 10), fonts, 0.5, color, 2)
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
                    cv.putText(image, 'WARNING!!!',
                               (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 7)
                    draw_text(image, label, cv.FONT_HERSHEY_PLAIN, (box[0], box[1] - 20), 2, 2, (54, 220, 209),
                              (0, 0, 255))
                    cv.rectangle(image, box, (0, 0, 255), 3)




# capture and call needed functions
def start_processing():
    camera = cv.VideoCapture('sample_videos/' + file_name)
    counter = 0
    capture = False
    width_zone = camera.get(3)
    height_zone = camera.get(4)
    # print("height:", height_zone, "width:", width_zone)
    x1 = int(width_zone / 2) - int(width_zone / 4)+200
    x2 = int(width_zone / 2) + int(width_zone / 4)-200
    x3 = int(width_zone) - 450
    x4 = 0 + 450
    y1 = int(height_zone - 50)
    y2 = int(height_zone - 5)
    ym = (y1 + y2) / 2

    while True:

        ret, frame = camera.read()

        pts = np.array([[x4, y2], [x1, y1], [x2, y1], [x3, y2]])
        cv.polylines(frame, [pts], True, (0, 255, 255), 2)

        orignal = frame.copy()

        ObjectDetector(frame)


        cv.imshow('frame', frame)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
    cv.destroyAllWindows()


yoloNet, model = setUpOpenCVNet();
start_processing()
