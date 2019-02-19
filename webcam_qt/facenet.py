from keras.models import load_model
import cv2
import os
import glob
import numpy as np
import face_recognition
import sys
import argparse
from utils import *

class Facenet():
    def __init__(self, detector, model_path = 'models/facenet_keras.h5', detector_model_path ='models/haarcascade_frontalface_default.xml'):
        self.detector_changed_soon = True
        self.face_cascade = None
        self.yoloutils = None
        self.detector = detector
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.detector_model_path = detector_model_path
        self.database = self.prepare_database()


    def change_model_path(self, model_path):
        self.model_path = model_path
    def change_detector_model_path(self, detector_model_path):
        self.detector_model_path = detector_model_path
        self.detector_changed_soon = True
    def change_detector(self, detector):
        self.detector = detector
        self.detector_changed_soon = True

    def euclidean_distance(self, image1, image2):
        distance = image1 - image2
        distance = np.sum(np.multiply(distance, distance))
        distance = np.sqrt(distance)
        return distance

    def encoded_image(self, image):
        image = cv2.resize(image, (160, 160))
        image = np.around(image / 255.0, decimals = 12)
        x_train = np.array([image])
        return self.model.predict(x_train)

    def prepare_database(self, ):
        database = {}
        for file in glob.glob("database/*"):
            identity = os.path.splitext(os.path.basename(file))[0]
            image = cv2.imread(file)

            face_image = self.extract_face_image(image)
            database[identity] = self.encoded_image(face_image)
        return database

    def extract_face_image(self, image):
        (x1, y1, x2, y2) = self.extract_face_coordinates(image)[0]
        return self.sub_image(image, x1, y1, x2, y2)

    def sub_image(self, image, x1, y1, x2, y2):
        height, width, channels = image.shape
        return image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    def extract_face_coordinates(self, image):
        face_coordinates = []

        if(self.detector == Facenet.Detectors.HAAR):
            if(self.detector_changed_soon):
                self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
                self.detector_changed_soon = False
            self.yoloutils = None
            if(not self.detector_model_path.split(".")[-1] == 'xml'):
                self.setDefaultModelPath()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            all_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x1, y1, w, h) in all_faces:
                x2 = x1 + w
                y2 = y1 + h
                face_coordinates.append([x1, y1, x2, y2])
        elif(self.detector == Facenet.Detectors.DLIB):
            self.detector_changed_soon = False
            self.yoloutils = None
            all_faces = face_recognition.face_locations(image)
            for (y1, x2, y2, x1) in all_faces:
                face_coordinates.append([x1, y1, x2, y2])
        elif(self.detector == Facenet.Detectors.YOLO):
            if (not self.detector_model_path.split(".")[-1] == 'weights'):
                self.setDefaultModelPath()
            if(self.detector_changed_soon):
                self.yoloutils = Facenet.YoloFaceUtils_CPU()
                self.detector_changed_soon = False
            all_faces = self.yoloutils.predict_yoloface(image)
            for (x1, y1, w, h) in all_faces:
                x2 = x1 + w
                y2 = y1 + h
                face_coordinates.append([x1, y1, x2, y2])

        return face_coordinates

    def setDefaultModelPath(self):
        if(self.detector  == Facenet.Detectors.YOLO):
            self.detector_model_path = 'models/yolov3-wider_16000.weights'
        elif(self.detector == Facenet.Detectors.DLIB):
            self.detector_model_path = 'models/haarcascade_frontalface_default.xml'
    def recognize_still_image(self, image):
        identities = []
        annotated_image = image.copy()
        for (x1, y1, x2, y2) in self.extract_face_coordinates(image):
            identity = self.find_identity(image, x1, y1, x2, y2)
            if identity is not None:
                annotated_image = self.annotate_image(annotated_image, identity, x1, y1, x2, y2)
                identities.append(identity)

        return annotated_image

    def annotate_image(self, image, identity, x1, y1, x2, y2):
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, identity.split(".")[0], (x1, y1), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def find_identity(self, image, x1, y1, x2, y2):
        face_image = self.sub_image(image, x1, y1, x2, y2)
        encoding = self.encoded_image(face_image)

        min_dist = 100
        identity = None
        for (name, db_enc) in self.database.items():
            dist = self.euclidean_distance(db_enc, encoding)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 8:
            return None
        else:
            print('Detected %s' %(identity))
            return str(identity)

    class Detectors():
            HAAR = 'HAAR'
            YOLO = 'Yolo'
            DLIB = 'DLIB'
    class YoloFaceUtils_CPU():
        def __init__(self, model_path="models/yolov3-wider_16000.weights", model_cfg="cfg/yolov3-face.cfg" ):
            self.model_path = model_path
            self.model_cfg = model_cfg
            self.net = self.load_yoloface()

        def load_yoloface(self):
            net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net

        def get_args(self,):
            parser = argparse.ArgumentParser()
            parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                                help='path to config file')
            parser.add_argument('--model-weights', type=str,
                                default='./models/yolov3-wider_16000.weights',
                                help='path to weights of model')
            parser.add_argument('--image', type=str, default='',
                                help='path to image file')
            parser.add_argument('--video', type=str, default='',
                                help='path to video file')
            parser.add_argument('--src', type=int, default=0,
                                help='source of the camera')
            parser.add_argument('--output-dir', type=str, default='outputs/',
                                help='path to the output directory')
            args = parser.parse_args()
            return args

        def predict_yoloface(self, frame):
            # IMG_HEIGHT, IMG_WIDTH, channels = frame.shape
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(get_outputs_names(self.net))

            # Remove the bounding boxes with low confidence
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            return faces

if __name__ == "__main__":
    facenet = Facenet(Facenet.Detectors.HAAR)
    cap = cv2.VideoCapture(0)
    while(True):

        ret, frame = cap.read()
        output = facenet.recognize_still_image(frame)
        cv2.imshow('image',output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
