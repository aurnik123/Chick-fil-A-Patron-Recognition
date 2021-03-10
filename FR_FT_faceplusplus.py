# Import the OpenCV and dlib libraries
import cv2
import os
import numpy as np
import uuid

import dlib
import PIL.Image
from Max_face_plus_plus.FE_branch import *
from classification import Inference

from openCV_facialrecog import *
import scipy.misc
import random
import string as sss

from cv2 import data

import threading
import time

import firebase_admin
from firebase_admin import credentials, firestore
from database import Customer
import time
import requests


fe = FeatureExtraction()
inf = Inference()
# Initialize a face cascade using the frontal face haar cascade provided with
# the OpenCV library
# Make sure that you copy this file from the opencv project to the root of this
# project folder
faceCascade = cv2.CascadeClassifier('./opencv-files/haarcascade_frontalface_default.xml')

# The desired output width and height
OUTPUT_SIZE_WIDTH = 960
OUTPUT_SIZE_HEIGHT = 720
CONFIDENCE_CUTOFF = 1000
RUN = True


# We are not doing really face recognition
def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[fid] = "Person " + str(fid)


block = False


def detectAndTrackMultipleFaces():
    global block
    # Open the first webcam device
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)
    #capture.set(cv2.CAP_PROP_EXPOSURE, 0.01)

    # Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    cv2.startWindowThread()

    # The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 0)

    # variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    # Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
    faceFeatures = {}

    cur_photo = None

    #####################################
    # setting up database
    cred = credentials.Certificate("serviceAccountKey.json")
    #firebase_admin.initialize_app(cred)
    db = firestore.client()

    col_ref = db.collection(u'Expo_Customers').order_by(u'last_updated').get()
    subjects = [""]
    confidence_value = 200

    for doc in col_ref:
        # print(doc.to_dict()['face_id'])
        subjects = subjects + [doc.to_dict()['face_id']]

    #####################################

    ###############################
    # importing and training facial recognizer
    subjects = ["", "Max", "Joel", "Sam"]
    faces, labels = prepare_training_data(r"expo_training")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    ################################

    ######################################
    #putting in training images
    number_subjects = 0
    dirs = os.listdir(r"expo_training")
    for dir_name in dirs:
        number_subjects = number_subjects + 1

    ######################################
    predict_final = None


    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()

            #fullSizeBaseImage = de_fisheye(fullSizeBaseImage)

            cur_photo = fullSizeBaseImage

            # Resize the image to 320x240
            baseImage = cv2.resize(fullSizeBaseImage, (960, 720))

            # Check if a key was pressed and if it was Q, then break
            # from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break

            # Result image is the image we will show the user, which is a
            # combination of the original image from the webcam and the
            # overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            # STEPS:
            # * Update all trackers and remove the ones that are not
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces.
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id

            # Increase the framecounter
            frameCounter += 1

            # Update all the trackers and remove the ones for which the update
            # indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[fid].update(baseImage)

                # If the tracking quality is good enough, we must delete
                # this tracker
                if trackingQuality < 7:
                    fidsToDelete.append(fid)

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop(fid, None)

            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 10) == 0:

                # For the face detection, we need to make use of a gray
                # colored image so we will convert the baseImage to a
                # gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)

                # Now use the haar cascade detector to find all faces
                # in the image
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                # Loop over all faces and check if the area for this
                # face is the largest so far
                # We need to convert it to int here because of the
                # requirement of the dlib tracker. If we omit the cast to
                # int here, you will get cast errors since the detector
                # returns numpy.int32 and the tracker requires an int
                for (_x, _y, _w, _h) in faces:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    # calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    # Variable holding information which faceid we
                    # matched with
                    matchedFid = None

                    # Now loop over all the trackers and check if the
                    # centerpoint of the face is within the box of a
                    # tracker
                    for fid in faceTrackers.keys():

                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        # check if the centerpoint of the face is within the
                        # rectangleof a tracker region. Also, the centerpoint
                        # of the tracker region must be within the region
                        # detected as a face. If both of these conditions hold
                        # we have a match
                        if ((t_x <= x_bar <= (t_x + t_w)) and
                                (t_y <= y_bar <= (t_y + t_h)) and
                                (x <= t_x_bar <= (x + w)) and
                                (y <= t_y_bar <= (y + h))):
                            matchedFid = fid

                    # If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        # TODO
                        print("Creating new tracker " + str(currentFaceID))
                        print('call')
                        cv2.imwrite('./Max_face_plus_plus/imgs/frame.jpg', cur_photo)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                        features = fe.parse_frame(cur_photo)
                        print(features)

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle(x - 10,
                                                           y - 20,
                                                           x + w + 10,
                                                           y + h + 20))

                        faceTrackers[currentFaceID] = tracker
                        mini_f = features
                        if type(features) == list and len(features) > 0:
                            mini = float('inf')
                            _x_ = x_bar / baseImage.shape[1]
                            _y_ = y_bar / baseImage.shape[0]
                            for ft in features:
                                loc_x = ft['face_rectangle']['left'] / cur_photo.shape[1]
                                loc_y = ft['face_rectangle']['top'] / cur_photo.shape[0]
                                if abs(loc_x - _x_) + abs(loc_y - _y_) < mini:
                                    mini = abs(loc_x - _x_) + abs(loc_y - _y_)
                                    mini_f = ft

                        faceFeatures[currentFaceID] = mini_f

                        if type(features) == list and len(features) > 0:
                            click = features[0]
                        else:
                            click = features
                        RUN = True
                        if type(mini_f) == list and len(mini_f) > 0:
                            click = mini_f[0]
                        elif click == None:
                            click = mini_f
                        if click != None:
                            try:
                                inf_inp = []
                                inf_inp.append(click['attributes']['age']['value'])
                                inf_inp.append(click['attributes']['gender']['value'].lower())
                                if click['attributes']['ethnicity']['value'].lower() == 'asian':
                                    inf_inp.append('asia')
                                else:
                                    inf_inp.append(click['attributes']['ethnicity']['value'].lower())
                                predict_final = inf.predict(inf_inp)
                                print(predict_final)
                            except:
                                print('something happened...')

                        # Start a new thread that is used to simulate
                        # face recognition. This is not yet implemented in this
                        # version :)
                        t = threading.Thread(target=doRecognizePerson,
                                             args=(faceNames, currentFaceID))
                        t.start()

                        # Increase the currentFaceID counter
                        currentFaceID += 1

            # Now loop over all the trackers we have and draw the rectangle
            # around the detected faces. If we 'know' the name for this person
            # (i.e. the recognition thread is finished), we print the name
            # of the person, otherwise the message indicating we are detecting
            # the name of the person
            for fid in faceTrackers.keys():
                tracked_position = faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                              (t_x + t_w, t_y + t_h),
                              rectangleColor, 2)


                ##########################################
                path = 'expo_training/s' + str(fid + number_subjects + 1)
                if (os.path.exists(path) == False):

                    os.mkdir(path)

                faceImage = baseImage[t_y:t_y + t_w, t_x:t_x + t_h]

                training_images = 0
                try:
                    for images in os.listdir(path):
                        training_images = training_images + 1

                    training_image_path = path + '/' + str(training_images + 1) + '.jpg'
                    if (training_images > 50 | number_subjects > 10):
                        pass
                    else:
                        scipy.misc.toimage(faceImage, cmin=0.0, cmax=255.0).save(training_image_path)

                    if (training_images == 1):
                        final_image = cv2.imread(training_image_path) #training_image_path
                        face = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                        #cv2.imshow("face", face)
                        label, confidence_value = face_recognizer.predict(face)
                        #label = face_recognizer.predict("expo/s1/24.jpg")
                except:
                    pass


                    #print(subjects[label[0]])
                ##########################################


                if fid in faceNames.keys():
                    if faceFeatures[fid] == None and not block:
                        print('call')
                        block = True
                        rc, fullSizeBaseImage = capture.read()

                        #fullSizeBaseImage = de_fisheye(fullSizeBaseImage)

                        features = FeatureExtraction().parse_frame(fullSizeBaseImage)
                        faceFeatures[fid] = features
                    counter = 0
                    if faceFeatures[fid] != None:
                        base = None
                        if type(faceFeatures[fid]) == list and len(faceFeatures[fid]) > 0:
                            base = faceFeatures[fid][0]['attributes']
                        elif type(faceFeatures[fid]) == dict:
                            base = faceFeatures[fid]['attributes']
                        # print(base)
                        
                        if base != None:
                            counter = 0
                            for key in base.keys():
                                string = ""
                                string += key + ' : '
                                string += str(base[key]['value'])
                                string += ','
                                cv2.putText(resultImage, string,
                                            (int(t_x + t_w / 2), int(t_y - counter * 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 255, 255), 1)
                                counter += 1

                        if(RUN):
                            try:
                                RUN = False
                                name = ''.join([random.choice(sss.ascii_letters + sss.digits) for _ in range(16)])
                                print('before')
                                threading.Thread(target=requests.get, args=(
                                'http://143.215.111.11:8000/getName?name=' + name + '&age=' + str(
                                    base['age']['value']) + '&gender=' + base['gender']['value'] + '&ethnicity=' +
                                base['ethnicity']['value'],)).start()
                            except Exception as e:
                                print(e)
                        ##################################
                        if (confidence_value < CONFIDENCE_CUTOFF): #customer is recognized
                            string = subjects[label]

                            cv2.putText(resultImage, string, (int(t_x + t_w / 2), int(t_y - counter * 10 - 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 1)

                            string = str(confidence_value)
                            cv2.putText(resultImage, string, (int(t_x + t_w / 2), int(t_y - counter * 10 - 30)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 255, 255), 1)

                            #r = requests.get('http://143.215.111.11:8000/getName?name=' + subjects[label])
                            #doc_rec = db.collection(u'Expo_Customers').where(u'face_id', u'==', str(subjects[label]))
                            #for doc in doc_rec:
                                #doc.update({u'inLine': True})
                                #doc.update({u'last_updated': int(round(time.time() * 1000))})


                        else:
                            unique_id = uuid.uuid4()
                            cv2.putText(resultImage, "Not Recognized",
                                        (int(t_x + t_w / 2), int(t_y - counter * 10 - 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 120, 120), 1)


                            #expo_col = db.collection(u'Expo_Customers')
                            #expo_col.document(str(unique_id)).set(
                                #Customer(.to_dict())



                        ##################################

                        # cv2.putText(resultImage, faceNames[fid] ,
                        #             (int(t_x + t_w/2), int(t_y)),
                        #             cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.5, (255, 255, 255), 2)


                else:
                    cv2.putText(resultImage, "Detecting...",
                                (int(t_x + t_w / 2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 165, 0), 1)

            # Since we want to show something larger on the screen than the
            # original 320x240, we resize the image again
            #
            # Note that it would also be possible to keep the large version
            # of the baseimage and make the result image a copy of this large
            # base image and use the scaling factor to draw the rectangle
            # at the right coordinates.
            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            # Finally, we want to show the images on the screen
            # cv2.imshow("base-image", baseImage)
            # cv2.imshow("base-image", resultImage)
            cv2.imshow("result-image", largeResult)




    # To ensure we can also deal with the user pressing Ctrl-C in the console
    # we have to check for the KeyboardInterrupt exception and break out of
    # the main loop
    except KeyboardInterrupt as e:
        pass

    # Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)


def format_face(features):
    base = features[0]['attributes']
    print(base['gender']['value'])

    for key in base.keys():
        string = ""
        string += key + ' : '
        string += str(base[key]['value'])
        string += ','
        #cv2.putText(resultImage, format_face(faceFeatures[fid]),
                    #(0, int(t_y)),
                    #cv2.FONT_HERSHEY_SIMPLEX,
                    #0.5, (255, 255, 255), 1)

def de_fisheye(nd_array):
    DIM = (640, 480)
    K = np.array([[329.0, 0.0, 422.0], [0.0, 329.0, 266.0], [0.0, 0.0, 1.0]])
    D = np.array([[0.04], [0.0011], [0.01], [-0.006]])

    h, w = nd_array.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(nd_array, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


if __name__ == '__main__':
    detectAndTrackMultipleFaces()
