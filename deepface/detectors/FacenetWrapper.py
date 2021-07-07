import cv2
from facenet_pytorch import MTCNN
import torch


def build_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device
    )
    return face_detector


def detect_face(face_detector, img, align=False):
    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
    delta = 0.5
    frame = img_rgb.copy()
    frame = cv2.resize(frame, (int(img.shape[1] * delta), int(img.shape[0] * delta)), interpolation=cv2.INTER_AREA)
    detections,conf = face_detector.detect(frame)
    if conf[0] != None:
        for (x, y, w, h) in detections:
            #if (w - x) > 40 * delta:
                x, y, w, h = int(x/delta), int(y/delta), int(w/delta), int(h/delta)
                detected_face = img[int(y):int(h), int(x):int(w)]
                img_region = [x, y, w-x, h-y]

    return detected_face, img_region

