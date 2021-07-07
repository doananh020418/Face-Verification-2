from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import time
import glob
from tqdm.notebook import tqdm
from deepface.commons import functions

device = 'cuda' if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)
def register(vid):
    scale = 0.2
    ptime = 0
    cap = cv2.VideoCapture(vid)
    count = 0
    frame_count = 0
    #while frame_count < 10:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_count = 0
        if ret:
            if  count % 1 == 0:
                #frame = cv2.resize(frame, (600, 400))

                img = frame.copy()
                img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation = cv2.INTER_AREA)
                # Here we are going to use the facenet detector
                boxes, conf = mtcnn.detect(img)

                # If there is no confidence that in the frame is a face, don't draw a rectangle around it
                if conf[0] != None:
                    for (x, y, w, h) in boxes:
                        if (w - x) > 40*scale:
                            text = f"{conf[0] * 100:.2f}%"
                            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            # detected_face = frame[int(y):int(h), int(x):int(w)]
                            # cv2.imwrite("static/frame/frame%d.jpg" % count, detected_face)
                            # print("frame %d saved" % count)
                            frame_count = frame_count + 1
                            cv2.putText(frame, text, (x, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 255), 1)

        count = count +1
        # Show the result
        # If we were using Google Colab we would use their function cv2_imshow()

        # For displaying images/frames
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

register(0)