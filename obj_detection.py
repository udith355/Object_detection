import cv2 as cv
import imutils
from imutils import paths
import numpy as np
import numpy as np

# import distance_to_camera as dc
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
        # print("score : ",score)
        label = "%s" % (class_names[classid])

        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 10), fonts, 0.5, color, 2)
    # ===========================================================================================
    marker = find_marker(image)
    KNOWN_DISTANCE = 16.0
    KNOWN_WIDTH = 6.0
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    print("Focal Length : ", focalLength)
    # =======================================================================================================
    for i, b in enumerate(boxes):
        print(boxes)

        if classes[i] == 2 or classes[i] == 7 or classes[i] == 8:
            if scores[i] >= 0.6:
                apx_distance = distance_to_camera(KNOWN_WIDTH, focalLength, abs(boxes[i][1] - boxes[i][3]))
                print("Distance : ", apx_distance)
                mid_x = (boxes[i][1] + boxes[i][3]) / 2

                mid_y = (boxes[i][0] + boxes[i][2]) / 2

                apx_distance = round(((1 - (boxes[i][3] - boxes[i][1]))), 1)
                print("print 1:", ((boxes[i][1])))
                print("print 2", ((boxes[i][3])))
                print("print 3:", ((boxes[i][3] - boxes[i][1])))
                print("print 4:", (1 - (boxes[i][3] - boxes[i][1])))
                print("apro distance: ", apx_distance)
                # cv.putText(image, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if (apx_distance) <= 350:
                if (mid_x) > (100) and (mid_x) < (400):
                    cv.putText(image, 'WARNING!!! {apx_distance}'.format(apx_distance=apx_distance),
                               (box[0] + 10, box[1] + 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv.rectangle(image, box, (0, 0, 255), 2)
            # if (apx_distance) <=50:
            #        if (mid_x) >(100)  and (mid_x) < (400):
            #            cv.putText(image, 'WARNING!!! {apx_distance}'.format(apx_distance = apx_distance),(box[0]+10, box[1]+50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            #            cv.rectangle(image, box, (0,0,255), 2)
    # return   boxes,classes,scores,image


# Detect distance of the objects
def Detect_Distance(boxes, classes, scores, image_np):
    for i, b in enumerate(boxes):
        if classes[i] == 2 or classes[i] == 7 or classes[i] == 8:
            if scores[i] >= 0.5:
                # print("type :",type(boxes[0][1]))
                mid_x = (boxes[i][1] + boxes[i][3]) / 2
                print("Mid_x", mid_x / 1000)

                mid_y = (boxes[i][0] + boxes[i][2]) / 2
                print("Mid_y", mid_y / 1000)
                apx_distance = round(((1 - (boxes[i][3] - boxes[i][1]))), 1)
                print("apro distance: ", apx_distance)
                cv.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if (apx_distance / 1000) <= 0.9:
                    if (mid_x / 1000) > 0.1 and (mid_x / 1000) < 0.9:
                        cv.putText(image_np, 'WARNING!!!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


# capture and call needed functions
def start_processing():
    camera = cv.VideoCapture('sample_videos/' + file_name)
    counter = 0
    capture = False
    # number = 0
    while True:
        ret, frame = camera.read()

        orignal = frame.copy()
        # DetectObject=
        ObjectDetector(frame)
        # Detect_Distance(DetectObject[0],DetectObject[1],DetectObject[2],DetectObject[3])
        # print(DetectObject[0])
        # print(DetectObject[0][0])

        # cv.imshow('oringal', orignal)

        # # print(capture == True and counter < 10)
        # if capture == True and counter < 10:
        #     counter += 1
        #     # cv.putText(
        #     #     frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
        # else:
        #     counter = 0

        cv.imshow('frame', frame)
        key = cv.waitKey(1)

        # if key == ord('c'):
        #     capture = True
        #     number += 1
        #     cv.imwrite(f'ReferenceImages/image{number}.png', orignal)
        if key == ord('q'):
            break
    cv.destroyAllWindows()


yoloNet, model = setUpOpenCVNet();
start_processing()
