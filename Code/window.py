# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!



import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 140, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.imageDisplay = QtWidgets.QLabel(self.centralwidget)
        self.imageDisplay.setGeometry(QtCore.QRect(100, 80, 461, 211))
        self.imageDisplay.setText("")
        self.imageDisplay.setObjectName("imageDisplay")
        self.outputDisplay = QtWidgets.QLabel(self.centralwidget)
        self.outputDisplay.setGeometry(QtCore.QRect(60, 370, 481, 171))
        self.outputDisplay.setText("")
        self.outputDisplay.setObjectName("outputDisplay")
        self.outputTitle = QtWidgets.QLabel(self.centralwidget)
        self.outputTitle.setGeometry(QtCore.QRect(110, 320, 101, 16))
        self.outputTitle.setObjectName("outputTitle")
        self.imageTitle = QtWidgets.QLabel(self.centralwidget)
        self.imageTitle.setGeometry(QtCore.QRect(110, 30, 91, 16))
        self.imageTitle.setObjectName("imageTitle")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.buttons()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Upload Image"))
        self.outputTitle.setText(_translate("MainWindow", "Output Equation"))
        self.imageTitle.setText(_translate("MainWindow", "Input Image"))
    def buttons(self):
        self.pushButton.clicked.connect(self.openImage)

    def openImage(self):
     path, par = QFileDialog.getOpenFileName(None, 'Open File')
     self.imageDisplay.setPixmap(QPixmap(path))
     self.imageDisplay.setScaledContents(True)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
