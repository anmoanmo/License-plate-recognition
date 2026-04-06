# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainui.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QWidget)
try:
    from . import img_rc
except ImportError:
    import img_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 592)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"")
        self.label_input = QLabel(self.centralwidget)
        self.label_input.setObjectName(u"label_input")
        self.label_input.setGeometry(QRect(40, 100, 512, 401))
        self.label_input.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(0, 0, 1001, 51))
        font = QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"background-color:rgb(80, 80, 80);\n"
"\n"
"color: rgb(255, 255, 255);")
        self.label_2.setTextFormat(Qt.AutoText)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_chepai = QLabel(self.centralwidget)
        self.label_chepai.setObjectName(u"label_chepai")
        self.label_chepai.setGeometry(QRect(650, 130, 245, 85))
        self.label_chepai.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.label_carNumber = QLabel(self.centralwidget)
        self.label_carNumber.setObjectName(u"label_carNumber")
        self.label_carNumber.setGeometry(QRect(650, 260, 245, 65))
        font1 = QFont()
        font1.setPointSize(17)
        self.label_carNumber.setFont(font1)
        self.label_carNumber.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.pushButton_UploadImg = QPushButton(self.centralwidget)
        self.pushButton_UploadImg.setObjectName(u"pushButton_UploadImg")
        self.pushButton_UploadImg.setGeometry(QRect(570, 440, 91, 41))
        self.pushButton_DetectPlate = QPushButton(self.centralwidget)
        self.pushButton_DetectPlate.setObjectName(u"pushButton_DetectPlate")
        self.pushButton_DetectPlate.setGeometry(QRect(680, 440, 91, 41))
        self.pushButton_RecognizePlate = QPushButton(self.centralwidget)
        self.pushButton_RecognizePlate.setObjectName(u"pushButton_RecognizePlate")
        self.pushButton_RecognizePlate.setGeometry(QRect(790, 440, 91, 41))
        self.pushButton_Clear = QPushButton(self.centralwidget)
        self.pushButton_Clear.setObjectName(u"pushButton_Clear")
        self.pushButton_Clear.setGeometry(QRect(900, 440, 91, 41))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_input.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u57fa\u4e8eYOLOv11+CNN\u7684\u8f66\u724c\u8bc6\u522b\u7cfb\u7edf", None))
        self.label_chepai.setText("")
        self.label_carNumber.setText("")
        self.pushButton_UploadImg.setText(QCoreApplication.translate("MainWindow", u" \u4e0a\u4f20\u56fe\u7247", None))
        self.pushButton_DetectPlate.setText(QCoreApplication.translate("MainWindow", u"\u8f66\u724c\u5b9a\u4f4d", None))
        self.pushButton_RecognizePlate.setText(QCoreApplication.translate("MainWindow", u"\u8bc6\u522b\u8f66\u724c", None))
        self.pushButton_Clear.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
    # retranslateUi

