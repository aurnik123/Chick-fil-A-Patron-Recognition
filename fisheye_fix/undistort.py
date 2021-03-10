import cv2
import numpy as np
import sys

# You should replace these 3 lines with the output in calibration step
DIM = (1920, 1080)
K = np.array([[1310.7672866100804, 0.0, 978.8739856522392], [-300.0, 1293.968763765934, 522.7650535309522],
    [-0.1, -0.1, 1.0]])
D = np.array(
    [[-0.3041451681795997], [0.44317577665404156], [-1.0544444640572284], [0.9697882377431465]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    undistort('20190403142459745.jpg')