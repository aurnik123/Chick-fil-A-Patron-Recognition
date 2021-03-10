import numpy as np
import cv2
from FR_branch import *
from FE_branch import *


class VideoProcessing():
    def __init__(self):
        self._fr = FacialRecognition()
        self._fe = FacialExtraction()

    def process_frame(self, frame):
        self._fr.parse_frame(frame)
        self._fe.parse_frame(frame)


# for USB/integrated cameras
cap = cv2.VideoCapture(0)

# for IP cameras
# cap = cv2.VideoCapture("http://192.168.18.37:8090/test.mjpeg")

vp = VideoProcessing()

if __name__ == '__main__':
    while True:
        # capture the image
        ret, frame = cap.read()

        # check for success
        if ret:
            print('Something Here')
            vp.process_frame(frame)
