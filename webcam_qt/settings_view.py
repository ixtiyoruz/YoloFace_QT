# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings.ui',
# licensing of 'settings.ui' applies.
#
# Created: Fri Feb 15 12:18:48 2019
#      by: pyside2-uic  running on PySide2 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *


class Ui_Settings(object):
    def setupUi(self, Settings):
        Settings.setObjectName("Settings")
        Settings.resize(338, 353)
        self.centralwidget = QWidget(Settings)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 70, 101, 16))
        self.label.setObjectName("label")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 60, 93, 41))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 130, 53, 16))
        self.label_2.setObjectName("label_2")
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(140, 130, 111, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        Settings.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Settings)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 338, 26))
        self.menubar.setObjectName("menubar")
        Settings.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Settings)
        self.statusbar.setObjectName("statusbar")
        Settings.setStatusBar(self.statusbar)

        self.retranslateUi(Settings)
        QtCore.QMetaObject.connectSlotsByName(Settings)

    def retranslateUi(self, Settings):
        Settings.setWindowTitle(QApplication.translate("Settings", "MainWindow", None, -1))
        self.label.setText(QApplication.translate("Settings", "Model", None, -1))
        self.pushButton.setText(QApplication.translate("Settings", "Select", None, -1))
        self.label_2.setText(QApplication.translate("Settings", "Detector", None, -1))
        self.comboBox.setItemText(0, QApplication.translate("Settings", "Yolo", None, -1))
        self.comboBox.setItemText(1, QApplication.translate("Settings", "HAAR", None, -1))
        self.comboBox.setItemText(2, QApplication.translate("Settings", "DLIB", None, -1))

