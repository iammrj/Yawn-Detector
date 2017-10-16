import cv2
import imutils
import numpy as np
import time
import sys

path = "classifiers/haar-face.xml"
faceCascade = cv2.CascadeClassifier(path)

# Variable used to hold the ratio of the contour area to the ROI
ratio = 0

# variable used to hold the average time duration of the yawn
global yawnStartTime
yawnStartTime = 0

# Flag for testing the start time of the yawn
global isFirstTime
isFirstTime = True

# List to hold yawn ratio count and timestamp
yawnRatioCount = []

# Yawn Counter
yawnCounter = 0

# yawn time
averageYawnTime = 2.5

"""
Find the second largest contour in the ROI; 
Largest is the contour of the bottom half of the face.
Second largest is the lips and mouth when yawning.
"""


def calculate_contours(image, contours):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    max_area = 0
    second_max = 0
    max_count = 0
    secondmax_count = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if max_area < area:
            second_max = max_area
            max_area = area
            secondmax_count = max_count
            max_count = count
        elif second_max < area:
            second_max = area
            secondmax_count = count

    return [secondmax_count, second_max]


"""
Thresholds the image and converts it to binary
"""


def threshold_contours(mouth_region, rect_area):
    global ratio

    # Histogram equalize the image after converting the image from one color space to another
    # Here, converted to grey scale
    img_ray = cv2.equalizeHist(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY))

    # Thresholding the image => outputs a binary image.
    # Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0.
    ret, thresh = cv2.threshold(img_ray, 64, 255, cv2.THRESH_BINARY)

    # Finds contours in a binary image
    # Constructs a tree like structure to hold the contours
    # Contouring is done by having the contoured region made by of small rectangles and storing only the end points
    # of the rectangle
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return_value = calculate_contours(mouth_region, contours)

    # return_value[0] => second_max_count
    # return_value[1] => Area of the contoured region.
    second_max_count = return_value[0]
    contour_area = return_value[1]

    ratio = contour_area / rect_area

    # Draw contours in the image passed. The contours are stored as vectors in the array.
    # -1 indicates the thickness of the contours. Change if needed.
    if isinstance(second_max_count, np.ndarray) and len(second_max_count) > 0:
        cv2.drawContours(mouth_region, [second_max_count], 0, (255, 0, 0), -1)


"""
Isolates the region of interest and detects if a yawn has occured. 
"""


def yawn_detector(video_capture):
    global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=450)
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Isolate the ROI as the mouth region
        width_one_corner = int((x + (w / 4)))
        width_other_corner = int(x + ((3 * w) / 4))
        height_one_corner = int(y + (11 * h / 16))
        height_other_corner = int(y + h)

        # Indicate the region of interest as the mouth by highlighting it in the window.
        cv2.rectangle(frame, (width_one_corner, height_one_corner), (width_other_corner, height_other_corner),
                      (0, 0, 255), 2)

        # mouth region
        mouth_region = frame[height_one_corner:height_other_corner, width_one_corner:width_other_corner]

        # Area of the bottom half of the face rectangle
        rect_area = (w * h) / 2

        if len(mouth_region) > 0:
            threshold_contours(mouth_region, rect_area)

        print("Current probability of yawn: " + str(round(ratio * 1000, 2)) + "%")
        print("Length of yawnCounter: " + str(len(yawnRatioCount)))

        if ratio > 0.06:
            if isFirstTime is True:
                isFirstTime = False
                yawnStartTime = time.time()

            # If the mouth is open for more than 2.5 seconds, classify it as a yawn
            if (time.time() - yawnStartTime) >= averageYawnTime:
                yawnCounter += 1
                yawnRatioCount.append(yawnCounter)

                if len(yawnRatioCount) > 30:
                    # Reset all variables
                    isFirstTime = True
                    yawnStartTime = 0
                    return True

    # Display the resulting frame
    cv2.namedWindow('yawnVideo')
    cv2.imshow('yawnVideo', frame)
    # time.sleep(0.025)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

    return False


def main():
    # Capture from web camera
    yawn_camera = cv2.VideoCapture("123.mp4")

    while True:
        return_value = (yawn_detector(yawn_camera), 'yawn')
        if return_value[0]:
            print("Yawn detected!")
            # When everything is done, release the capture
            yawn_camera.release()
            cv2.destroyWindow('yawnVideo')
            return return_value


main()