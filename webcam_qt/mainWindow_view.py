# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui',
# licensing of 'mainWindow.ui' applies.
#
# Created: Fri Feb 15 12:23:44 2019
#      by: pyside2-uic  running on PySide2 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore
from PyQt4.QtGui import *


class Ui_FaceDetector(object):
    def setupUi(self, FaceDetector):
        FaceDetector.setObjectName("FaceDetector")
        FaceDetector.resize(727, 588)
        self.centralwidget = QWidget(FaceDetector)
        self.centralwidget.setObjectName("centralwidget")
        self.videoFrame = QLabel(self.centralwidget)
        self.videoFrame.setGeometry(QtCore.QRect(10, 0, 711, 541))
        self.videoFrame.setFrameShape(QFrame.WinPanel)
        self.videoFrame.setFrameShadow(QFrame.Raised)
        self.videoFrame.setMidLineWidth(2)
        self.videoFrame.setObjectName("videoFrame")
        FaceDetector.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(FaceDetector)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 727, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuActions = QMenu(self.menubar)
        self.menuActions.setObjectName("menuActions")
        FaceDetector.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(FaceDetector)
        self.statusbar.setObjectName("statusbar")
        FaceDetector.setStatusBar(self.statusbar)
        self.actionVideo = QAction(FaceDetector)
        self.actionVideo.setObjectName("actionVideo")
        self.actionCamera = QAction(FaceDetector)
        self.actionCamera.setObjectName("actionCamera")
        self.actionModel = QAction(FaceDetector)
        self.actionModel.setObjectName("actionModel")
        self.actionDetector = QAction(FaceDetector)
        self.actionDetector.setObjectName("actionDetector")
        self.actionSettings = QAction(FaceDetector)
        self.actionSettings.setObjectName("actionSettings")
        self.menuFile.addAction(self.actionVideo)
        self.menuFile.addAction(self.actionCamera)
        self.menuActions.addAction(self.actionSettings)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuActions.menuAction())

        self.retranslateUi(FaceDetector)
        QtCore.QMetaObject.connectSlotsByName(FaceDetector)

    def retranslateUi(self, FaceDetector):
        FaceDetector.setWindowTitle(QApplication.translate("FaceDetector", "MainWindow", None, -1))
        self.videoFrame.setText(QApplication.translate("FaceDetector", "Video Frame", None, -1))
        self.menuFile.setTitle(QApplication.translate("FaceDetector", "File", None, -1))
        self.menuActions.setTitle(QApplication.translate("FaceDetector", "Actions", None, -1))
        self.actionVideo.setText(QApplication.translate("FaceDetector", "Video", None, -1))
        self.actionCamera.setText(QApplication.translate("FaceDetector", "Camera", None, -1))
        self.actionModel.setText(QApplication.translate("FaceDetector", "Settings", None, -1))
        self.actionDetector.setText(QApplication.translate("FaceDetector", "Detector", None, -1))
        self.actionSettings.setText(QApplication.translate("FaceDetector", "Settings", None, -1))
        self.actionSettings.setShortcut(QApplication.translate("FaceDetector", "Ctrl+S", None, -1))

