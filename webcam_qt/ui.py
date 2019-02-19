# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(727, 588)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.btnCapture = QtGui.QPushButton(self.centralwidget)
        self.btnCapture.setGeometry(QtCore.QRect(150, 300, 75, 23))
        self.btnCapture.setObjectName(_fromUtf8("btnCapture"))
        self.btnSave = QtGui.QPushButton(self.centralwidget)
        self.btnSave.setGeometry(QtCore.QRect(480, 490, 75, 23))
        self.btnSave.setObjectName(_fromUtf8("btnSave"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(500, 290, 91, 20))
        self.label.setObjectName(_fromUtf8("label"))
        self.videoFrame = QtGui.QLabel(self.centralwidget)
        self.videoFrame.setGeometry(QtCore.QRect(30, 30, 311, 229))
        self.videoFrame.setFrameShape(QtGui.QFrame.WinPanel)
        self.videoFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.videoFrame.setMidLineWidth(2)
        self.videoFrame.setObjectName(_fromUtf8("videoFrame"))
        self.videoFrame_2 = QtGui.QLabel(self.centralwidget)
        self.videoFrame_2.setGeometry(QtCore.QRect(370, 30, 311, 229))
        self.videoFrame_2.setFrameShape(QtGui.QFrame.WinPanel)
        self.videoFrame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.videoFrame_2.setMidLineWidth(2)
        self.videoFrame_2.setObjectName(_fromUtf8("videoFrame_2"))
        self.btnPrint = QtGui.QPushButton(self.centralwidget)
        self.btnPrint.setGeometry(QtCore.QRect(400, 490, 75, 23))
        self.btnPrint.setWhatsThis(_fromUtf8(""))
        self.btnPrint.setObjectName(_fromUtf8("btnPrint"))
        self.TextFieldAddText = QtGui.QLineEdit(self.centralwidget)
        self.TextFieldAddText.setGeometry(QtCore.QRect(420, 360, 201, 20))
        self.TextFieldAddText.setObjectName(_fromUtf8("TextFieldAddText"))
        self.btnAddText = QtGui.QPushButton(self.centralwidget)
        self.btnAddText.setGeometry(QtCore.QRect(490, 400, 75, 23))
        self.btnAddText.setWhatsThis(_fromUtf8(""))
        self.btnAddText.setObjectName(_fromUtf8("btnAddText"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 727, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.btnCapture.setText(_translate("MainWindow", "Capture", None))
        self.btnSave.setText(_translate("MainWindow", "Save", None))
        self.label.setText(_translate("MainWindow", "Captured Frame", None))
        self.videoFrame.setText(_translate("MainWindow", "Video Frame", None))
        self.videoFrame_2.setText(_translate("MainWindow", "Video Frame", None))
        self.btnPrint.setText(_translate("MainWindow", "Print", None))
        self.btnAddText.setText(_translate("MainWindow", "Add Text", None))

