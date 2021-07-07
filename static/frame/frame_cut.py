import cv2
import glob
import os
from deepface.commons import functions

def register(vid):
    vidcap = cv2.VideoCapture(vid)
    success, image = vidcap.read()
    count = 0
    frame_count = 0
    while success and frame_count < 10:
        if count % 5 == 0:
            img = functions.preprocess_face(image,enforce_detection=False)
            cv2.imwrite("frame%d.jpg" % count, img)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            frame_count  = frame_count + 1
        count += 1
register(0)
# path = r'C:\Users\doank\PycharmProjects\deepface\static\frame'
# files = glob.glob(path+'/*.jpg')
# for f in files:
#     os.remove(f)