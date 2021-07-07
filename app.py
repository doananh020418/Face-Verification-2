import os
import time

import cv2
import torch
from facenet_pytorch import MTCNN
from flask import Flask, Response, request, jsonify,render_template
from werkzeug.utils import secure_filename
import glob
from deepface import DeepFace
from deepface.Face_recognition_vjpro import add_employee, get_df
from deepface.commons import functions, distance as dst

base_df = get_df(os.path.abspath('static/hieu'))
app = Flask(__name__)
print('loading')
cap = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=False,
    device=device
)
# ALLOWED_EXTENSIONS = set(['doc','docx', 'pdf', 'png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS = set(['jpg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_frames(df):  # generate frame by frame from camera
    # print(len(df))
    scale = 0.25
    frame_threshold = 10
    time_threshold = 5
    delta = 0.8
    model_name = 'VGG-Face'
    model = DeepFace.build_model(model_name)
    distance_metric = 'euclidean_l2'
    input_shape = (224, 224)
    text_color = (255, 255, 255)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]
    pivot_img_size = 112

    freeze = False
    face_detected = False
    face_included_frames = 0
    freezed_frame = 0
    tic = time.time()
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if img is None:
            break

        threshold = dst.findThreshold(model_name, distance_metric)

        raw_img = img.copy()

        resolution_x = img.shape[1]
        resolution_y = img.shape[0]
        frame = img.copy()
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        if freeze == False:
            faces, conf = mtcnn.detect(frame)
            if conf[0] == None:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        if conf[0] != None:
            for (x, y, w, h) in faces:
                if (w - x) > 50 / 4:
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1  # increase frame for a single face
                    x, y, w, h = int(x / scale), int(y / scale), int((w - x) / scale), int((h - y) / scale)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 67, 67), 1)  # draw rectangle to main image

                    cv2.putText(img, str(frame_threshold - face_included_frames), (int(x + w / 4), int(y + h / 1.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                    # -------------------------------------

                    detected_faces.append((x, y, w, h))
                    face_index = face_index + 1

            # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze:

            toc = time.time()
            if (toc - tic) <= time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        cv2.rectangle(freeze_img, (x, y), (x + w, y + h), (255, 67, 67),
                                      1)

                        custom_face = base_img[y:y + h, x:x + w]

                        custom_face = functions.preprocess_face(img=custom_face,
                                                                target_size=(input_shape_y, input_shape_x),
                                                                enforce_detection=False, detector_backend='facenet')

                        if custom_face.shape[1:3] == input_shape:
                            if df.shape[0] > 0:
                                img1_representation = model.predict(custom_face)[0, :]

                                def findDistance(row):
                                    distance_metric = row['distance_metric']
                                    img2_representation = row['embedding']
                                    distance = 1000  # initialize very large value
                                    if distance_metric == 'cosine':
                                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean':
                                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean_l2':
                                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                                             dst.l2_normalize(img2_representation))

                                    return distance

                                df['distance'] = df.apply(findDistance, axis=1)
                                df = df.sort_values(by=["distance"])

                                candidate = df.iloc[0]
                                employee_name = candidate['employee']
                                best_distance = candidate['distance']

                                if best_distance <= threshold * delta:
                                    # print(employee_name)
                                    display_img = cv2.imread(employee_name)

                                    display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

                                    label = employee_name.split("\\")[-1].split('/')[0]

                                    try:
                                        if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                            # top right
                                            freeze_img[y - pivot_img_size:y, x + w:x + w + pivot_img_size] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x + w, y), (x + w + pivot_img_size, y + 20),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x + w, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y),
                                                     (x + 3 * int(w / 4), y - int(pivot_img_size / 2)), (255, 67, 67),
                                                     1)
                                            cv2.line(freeze_img, (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                                     (x + w, y - int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                            # bottom left
                                            freeze_img[y + h:y + h + pivot_img_size, x - pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x - pivot_img_size, y + h - 20), (x, y + h),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x - pivot_img_size, y + h - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y + h),
                                                     (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (x, y + h + int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                            # top left
                                            freeze_img[y - pivot_img_size:y, x - pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x - pivot_img_size, y), (x, y + 20),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x - pivot_img_size, y + 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y),
                                                     (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                                     (x, y - int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif x + w + pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                            # bottom righ
                                            freeze_img[y + h:y + h + pivot_img_size,
                                            x + w:x + w + pivot_img_size] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x + w, y + h - 20),
                                                          (x + w + pivot_img_size, y + h), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x + w, y + h - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y + h),
                                                     (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (x + w, y + h + int(pivot_img_size / 2)), (255, 67, 67), 1)
                                    except Exception as err:
                                        print(str(err))
                                    t = time.time()
                                else:
                                    t = toc - time_threshold

                        tic = time.time()

                    # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (255, 67, 67), -10)
                cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', freeze_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def home():
    ret = 'hello'
    return ret

@app.route('/<string:name>')
def streamimg(name):
    global df
    df = base_df[base_df['name']==name.strip()]
    return render_template('index.html')


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_file_api():
    if request.method == "POST":
        id = request.args.get('id')
        foldername = str(id)
        path = os.path.join(os.path.abspath('static'), foldername)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            files = glob.glob(path+'/*')
            for f in files:
                os.remove(f)
        # check if the post request has the file part
        if 'image' not in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400
            return resp
        files = request.files.getlist('image')
        errors = {}
        success = False
        file_names = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(path, file.filename))
                file_names.append(filename)
        tmp_df = add_employee(id, path)
        # base_df = base_df.append(tmp_df)
        ret = 'Update employee ' + str(id) + ' successful! \nFound ' + str(len(file_names)) + ' new imgs!'
        print('len tmp_df', len(tmp_df))
        print(id)
    sc = jsonify({'message': ret})
    sc.status_code = 200


    return sc


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(df), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
