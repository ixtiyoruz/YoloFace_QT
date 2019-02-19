import sys
import cv2
import numpy as np
from PyQt4 import QtGui, QtCore, Qt
from mainWindow_view import Ui_FaceDetector
from settings_view import Ui_Settings
import facenet

class Video():
    def __init__(self,capture, facenet):
        self.capture = capture
        self.currentFrame=np.array([])
        self.facenet= facenet
        # if you want to reduce frame rate, but it doesnt work as you expected
        self.fps_delay = 1000
        self.frms_count = 1

    def captureFrame(self):
        """
        capture frame and return captured frame
        """
        ret, readFrame = self.capture.read()
        if(self.frms_count % self.fps_delay == 0):
            newframe = self.facenet.recognize_still_image(readFrame)
        else:
            newframe = readFrame
        return newframe


    def captureNextFrame(self):
        """                           
        capture frame and reverse RBG BGR and return opencv image                                      
        """
        ret, readFrame=self.capture.read()
        newframe = self.facenet.recognize_still_image(readFrame)

        if(ret==True):
            self.currentFrame=cv2.cvtColor(newframe,cv2.COLOR_BGR2RGB)



    def convertFrame(self):
        """     converts frame to format suitable for QtGui            """
        try:
            height,width=self.currentFrame.shape[:2]
            img=QtGui.QImage(self.currentFrame,
                              width,
                              height,
                              QtGui.QImage.Format_RGB888)
            img=QtGui.QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except:
            return None

    def convertSpecifiedFrame(frame):
        """     converts frame to format suitable for QtGui            """
        try:
            height,width=frame.shape[:2]
            img=QtGui.QImage(frame,
                              width,
                              height,
                              QtGui.QImage.Format_RGB888)
            img=QtGui.QPixmap.fromImage(img)
            return img
        except:
            return None

    def getImage(self):
        return cv2.imread("test.jpg")
 
class Gui(QtGui.QMainWindow):
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_FaceDetector()
        self.ui.setupUi(self)
        self.filename = ""
        self.detector = facenet.Facenet.Detectors.YOLO
        self.facenet = facenet.Facenet(self.detector)
        self.video = Video(cv2.VideoCapture(0), self.facenet)
        self.settings = Settings()
        self.initui()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.play)
        self._timer.start(27)
        self.update()
        self.ret, self.capturedFrame = self.video.capture.read()

    def initui(self):
        # this is the action which we will put in the menu
        self.ui.actionVideo.setShortcut("Ctrl+O")
        self.ui.actionVideo.setStatusTip("Open a video")
        self.ui.actionVideo.triggered.connect(self.openFile)
        self.ui.actionCamera.setShortcut("Ctrl+C")
        self.ui.actionCamera.setStatusTip("Open the Camera")
        self.ui.actionCamera.triggered.connect(self.openCamera)
        self.ui.actionCamera.setShortcut("Ctrl+S")
        self.ui.actionCamera.setStatusTip("Settings")
        self.ui.actionSettings.triggered.connect(self.openSettings)
        self.settings.ui.comboBox.activated[str].connect(self.select_detector)
        self.settings.ui.pushButton.clicked.connect(self.openModel_path)

    def select_detector(self, text):
        print('detector changed to --', text)
        self.detector = text
        self.facenet.change_detector(self.detector)

    def openSettings(self):
        self.settings.show()

    def openFile(self):
        self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if(self.filename):
            self.video = Video(cv2.VideoCapture(self.filename), self.facenet)

    def openModel_path(self):
        text = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if(text):
            self.model_path = text
            self.facenet.change_model_path(self.model_path)
        print(self.model)

    def openCamera(self):
        self.video = Video(cv2.VideoCapture(0))

    def play(self):
        try:
            self.video.captureNextFrame()
            self.ui.videoFrame.setPixmap(self.video.convertFrame())
            self.ui.videoFrame.setScaledContents(True)
        except TypeError:
            print ("No frame")

        # this is an event for frame x button it takes all the closing eventis and triggers the close application function

    def closeEvent(self, event):
        self.settings.close()

class Settings(QtGui.QMainWindow):

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_Settings()
        self.ui.setupUi(self)

def main():
    app = QtGui.QApplication(sys.argv)
    ex = Gui()
    ex.show()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()