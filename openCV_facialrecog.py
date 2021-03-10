import cv2  # OpenCV
import os
import numpy as np

from skimage.transform import resize


'''print("testing")
# subjects = ["", "Elvis Presley", "Freddie Mercury"]
# subjects = ["", "Drew Teachout", "Jordan Leahey"]
subjects = ["", "Kris", "Chase", "Zach", "Paul", "Nikhil"]
# subjects = ["", "Kris", "Zach", "Paul", "Nikhil"]'''



# function to detect face using OpenCV
# MIGHT NOT NEED TO USE IF DIAN'S CODE PASSES IN CROPPED FACE
def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, local binary patterns classifier
    # OTHER OPTIONS: frontal vs. profile face (profile for edge cases), LBP vs Haar classifier (Haar is slower but more accurate)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface_improved.xml')
    # face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_profileface.xml')
    # face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')


    # detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if len(faces) == 0:
        return None, None

    # under the assumption that there will be only one face, extract face
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            # cv2.imshow("Training on image...", image)
            # cv2.waitKey(1)

            # detect face
            face, rect = detect_face(image)

            # ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                print(image_path)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# TRAINING DATA
'''print("Preparing data...")
faces, labels = prepare_training_data(r"expo_training")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))'''


# create our LBPH face recognizer
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# or use EigenFaceRecognizer by replacing above line with
'''face_recognizer = cv2.face.EigenFaceRecognizer_create()
faces1 = []
for x in faces:
    faces1.append(resize(x, (100, 100), anti_aliasing=True))

faces = faces1'''

# or use FisherFaceRecognizer by replacing above line with
'''face_recognizer = cv2.face.FisherFaceRecognizer_create()
faces1 = []
for x in faces:
    faces1.append(resize(x, (100, 100), anti_aliasing=True))

faces = faces1'''


# train our face recognizer of our training faces
#face_recognizer.train(faces, np.array(labels))
#face_recognizer.save("Dian_XML.xml")

# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):

    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    if face is not None:
        # COMMENT OUT FOR LBPH
        # face = resize(face, (100, 100), anti_aliasing=True)

        # predict the image using our face recognizer
        label = face_recognizer.predict(face)
        # get name of respective label returned by face recognizer
        label_text = subjects[label[0]]

        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


'''print("Predicting images...")

# RECOGNIZER HAS BEEN TRAINED WITH INPUT IMAGES, NOW FOR PREDICTION

# load test images
test_img1 = cv2.imread("3.12.19_test/7.jpg")
test_img2 = cv2.imread("3.12.19_test/23.jpg")
test_img3 = cv2.imread("3.12.19_test/43.jpg")
test_img4 = cv2.imread("3.12.19_test/49.jpg")
test_img5 = cv2.imread("3.12.19_test/70.jpg")

# perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
print("Prediction complete")

# display both images
cv2.imshow("Prediction 1", cv2.resize(predicted_img1, (400, 600)))
cv2.waitKey(0)
cv2.imshow("Prediction 2", cv2.resize(predicted_img2, (400, 600)))
cv2.waitKey(0)
cv2.imshow("Prediction 3", cv2.resize(predicted_img3, (400, 600)))
cv2.waitKey(0)
cv2.imshow("Prediction 4", cv2.resize(predicted_img4, (400, 600)))
cv2.waitKey(0)
cv2.imshow("Prediction 5", cv2.resize(predicted_img5, (400, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()'''
