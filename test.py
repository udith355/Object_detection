import cv2 as cv
import numpy as np

#
# input sample video file name
#
file_name = 'test5.mp4'

#
# Press Q to close the window
#

# Threshold values
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_SIMPLEX

camera = cv.VideoCapture('sample_videos/' + file_name)

width_zone = camera.get(3)
height_zone = camera.get(4)

# print("height:", height_zone, "width:", width_zone)
x1 = int(width_zone / 2) - int(width_zone / 4)
x2 = int(width_zone / 2) + int(width_zone / 4)
x3 = int(width_zone) - 20
x4 = 0 + 20
y1 = int(height_zone - 100)
y2 = int(height_zone)
ym = (y1 + y2) / 2

tpo = 0
tpr = 0

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


def roi(x, y, img):
    m1 = (y2 - y1) / (x4 - x1)
    m2 = (y2 - y1) / (x3 - x2)
    if (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= ym and y < y2):
        return 2
    elif (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= y1 and y < ym):
        return 1
    else:
        return 0


def ObjectDetector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s" % (class_names[classid])

        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 10), fonts, 0.5, color, 2)


yoloNet, model = setUpOpenCVNet()

c = 0
while True:
    if (c % 10 != 0):
        _, img = camera.read()

        img = cv.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        c += 1

        continue
    c += 1
    _, img = camera.read()

    img = cv.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    pts = np.array([[x4, y2], [x1, y1], [x2, y1], [x3, y2]])
    cv.polylines(img, [pts], True, (0, 255, 255), 2)

    blob = cv.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    yoloNet.setInput(blob)

    output_layers_names = yoloNet.getUnconnectedOutLayersNames()
    layerOutputs = yoloNet.forward(output_layers_names)

    boxes = []
    class_ids = []
    confidences = []

    ObjectDetector(img)

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # obj detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width + 20)
                h = int(detection[3] * height + 20)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if w * h < 200000:
                label = str(class_names[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if roi(x + w / 2, y + h, img) == 2:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(img, "Careful", (x, y + 20), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv.imshow('Image', img)
    # cv2.imshow('ROI', roi)
    key = cv.waitKey(200)
    if key == 27:
        break

camera.release()
cv.destroyAllWindows()

