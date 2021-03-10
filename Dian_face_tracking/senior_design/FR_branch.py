import numpy as np
import cv2

class FacialRecognition():
    def __init__(self):
        self.gray = None
        self.total_face_num = 0
        self.faces = None
        self.color_faces = []
        self.faceID = []

    def parse_frame(self, frame):
        pass

    def face_detect(self, frame):
        """
        Get the faces in frame

        try different cascades params

        :param frame: image
        :return: a list of images of faces
        """
        roi_color_faces = []
        # Multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
        #face_cascade = cv2.CascadeClassifier('../../opencv-files/lbpcascade_frontalface_improved.xml')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x,y,w,h) in self.faces:
            roi_color_faces.append(frame[y:y+h, x:x+w])
        self.total_face_num = len(self.faces)
        self.color_faces = roi_color_faces
        print("detected {} faces".format(self.total_face_num))
        return roi_color_faces

    # TODO
    def get_faceID(self):
        return 0
            
        
if __name__ == '__main__':
    fr = FacialRecognition()
    faces = fr.face_detect(cv2.imread("faceTest.jpg")) 
    for i,face in enumerate(faces):
        cv2.imwrite("face{}".format(i) + ".jpg", face)
        
        
