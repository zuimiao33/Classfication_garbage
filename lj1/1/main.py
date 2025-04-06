from re import S
import sys
import PyQt5
from PyQt5 import QtWidgets, QtGui, QtCore
# from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
# from PyQt5.QtCore import QTimer
from distutils.log import error
from webbrowser import get
import paddlex as pdx
import time
from AnimFunction import *


class model(QtCore.QThread):
    _display = pyqtSignal(int)
    _display1 = pyqtSignal(QImage)

    def __init__(self, model):
        super().__init__()
        self.model = model
        # 有害垃圾的数量（干电池（1 号、2 号、5 号）,药盒）
        self.class_youhai = ["Battery", "Cigarette", "Pillbox"]
        # 厨余垃圾的数量（小土豆、切过的白萝卜、胡萝卜，尺寸为电池大小,蔬菜 ,白萝卜）
        self.class_chuyu = ["Potato", "White_radish", "Carrot", "Vegetable", "WhiteRadish"]
        # 可回收垃圾的数量（易拉罐、小号矿泉水瓶）
        self.class_huishou = ["Cans", "Bottle"]
        # 其他垃圾的数量（瓷片、鹅卵石（小土豆大小））
        self.class_qita = ["Ceramics", "Pebbles"]
        # 检测结果
        self.dect = ""
        # 置信度
        self.rate = 0
        # 检测是否成功的标志,0代表成功，1代表失败
        self.flag_1 = 0
        # flag 0 表示开始 flag 1 代表结束
        self.flag = 0
        self.x = 0
        self.frame1 = []

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame1 = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            self.frame1 = frame1
            self._display1.emit(self.frame1)
            if ret != True:
                break
            # cv2.imshow('video', frame)
            frame = frame.astype('float32')
            result = self.model.predict(frame)
            if self.model.model_type == "classifier":
                dect = result[0]['category']
                rate = result[0]['score']
                if dect in self.class_youhai and rate >= 0.7:
                    self.flag = 0
                    self.x = 1
                    self._display.emit(self.x)
                elif dect in self.class_huishou and rate >= 0.7:
                    self.flag = 0
                    self.x = 2
                    self._display.emit(self.x)
                elif dect in self.class_chuyu and rate >= 0.7:
                    self.flag = 0
                    self.x = 3
                    self._display.emit(self.x)
                elif dect in self.class_qita and rate >= 0.7:
                    self.flag = 0
                    self.x = 4
                    self._display.emit(self.x)
                else:
                    self.flag = 1
                if self.flag == 0:
                    print(result)
                    # time.sleep(1)
                if self.flag == 1:
                    time.sleep(0.1)

                    # 等待键盘事件，如果为q，退出
            key = cv2.waitKey(10)
            if key & 0xff == ord('q'):
                break

            # 等待键盘事件，如果为k,重置计数
            key = cv2.waitKey(10)

        cap.release()
        cv2.destroyAllWindows()


class Designer(QMainWindow):
    def __init__(self):
        super().__init__()
        # 公用数据
        self.w = 0
        self.h = 0
        self.y = 0
        self.m = 0
        self.i = 0
        self.o = 0

        self.frame_counter = 0
        # 存图片
        self.frame = []
        # 检测flag
        self.detectFlag = False
        self.cap = []
        # 定义定时器
        self.timer_camera = QTimer()
        self.model = pdx.load_model('11')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setupUi(self)
        #       光影效果
        UiFunction.Shaow(self)
        self.thread1 = model(self.model)
        self.thread1._display.connect(self.updateobjname)
        self.thread1._display1.connect(self.showimg)
        self.thread1.start()

    def Meau(self):
        UiFunction.MeauFunction(self)

    def Exit(self):
        sys.exit(app.exec_())

    #       重写鼠标移动事件
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True

    def closeEvent(self, event):
        super().closeEvent()
        self.camera.release()
        cv2.destroyAllWindows()

    def openFrame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            self.frame_counter += 1
            if self.frame_counter == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_counter = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                pass
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = frame.shape
            # 每行像素总数
            bytesPerLine = bytesPerComponent * width
            # （获取图像首地址，宽，高，存入格式）。自适应缩放
            q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888). \
                scaled(self.label.width(), self.label.height())
            # 绘制
            self.label.setPixmap(QPixmap.fromImage(q_image))

    def updateobjname(self, int):
        self.label_13.setObjectName(str(int))

    def showimg(self, img):
        img = img.scaled(self.label_16.width(), self.label_16.height())
        self.label_16.setPixmap(QPixmap.fromImage(img))

    def setupUi(self, MainWindow):
        MainWindow.resize(2400, 1400)

        self.main_widget = QtWidgets.QWidget(MainWindow)
        self.main_widget.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0.224, y1:0.705, x2:1, y2:0, stop:0 rgba(0, 192, 202, "
            "243), stop:1 rgba(53, 53, 209, 240));\n "
            "border-radius:40px;")

        self.Page_Widget = QtWidgets.QStackedWidget(self.main_widget)
        self.Page_Widget.setGeometry(QtCore.QRect(180, 40, 1280, 720))
        self.Page_Widget.setStyleSheet("background-color:None;\n"
                                       "border-radius:20px;")

        self.page_one = QtWidgets.QWidget()

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(1100, 50, 360, 100))
        self.label_2.setStyleSheet("background-color:None;\n"
                                   "border-radius:20px; font-size:50px;")
        self.w = 0
        self.label_2.setText("有害垃圾: %d" % self.w)

        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(1100, 180, 360, 100))
        self.label_7.setStyleSheet("background-color:None;\n"
                                   "border-radius:20px; font-size:50px;")
        self.label_7.setText("可回收垃圾: %d" % self.h)

        self.label_11 = QtWidgets.QLabel(self)
        self.label_11.setGeometry(QtCore.QRect(1100, 310, 360, 100))
        self.label_11.setStyleSheet("background-color:None;\n"
                                    "border-radius:20px;font-size:50px;")
        self.label_11.setText("厨余垃圾: %d" % self.y)

        self.label_12 = QtWidgets.QLabel(self)
        self.label_12.setGeometry(QtCore.QRect(1100, 440, 360, 100))
        self.label_12.setStyleSheet("background-color:None;\n"
                                    "border-radius:20px;font-size:50px;")
        self.label_12.setText("其余垃圾: %d" % self.m)

        self.label_13 = QtWidgets.QLabel(self)
        self.label_13.setText(" 垃圾种类")
        self.label_13.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, "
                                    "stop:0 rgba(0, 0, "
                                    "0, 76), "
                                    "stop:1 rgba(255, 255, 255, 63)); font-size:75px;border-radius:30px ")
        self.label_13.resize(450, 480)
        self.label_13.move(1450, 40)
        self.label_13.setObjectName("0")
        self.n = self.label_13.objectName()

        self.label_16 = QtWidgets.QLabel(self)
        self.label_16.resize(800, 500)
        self.label_16.move(1100, 550)
        self.label_16.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, "
                                    "stop:0 rgba(0, 0, "
                                    "0, 76), "
                                    "stop:1 rgba(255, 255, 255, 63)); font-size:75px;border-radius:30px ")

        self.btn_w = QtWidgets.QPushButton(self)

        def cao6():
            if self.label_13.objectName() == "1":
                self.label_13.setText("有害垃圾")
                self.w += 1
                self.label_name.setObjectName("1")
                self.label_13.setObjectName("0")
                self.label_name.setObjectName("0")
                self.label_2.setText("有害垃圾:%d" % self.w)
            elif self.label_13.objectName() == "2":
                self.label_13.setText("可回收垃圾")
                self.h += 1
                self.label_name.setObjectName("2")
                self.label_13.setObjectName("0")
                self.label_name.setObjectName("0")
                self.label_7.setText("可回收垃圾:%d" % self.h)
            elif self.label_13.objectName() == "3":
                self.label_13.setText("厨余垃圾")
                self.y += 1
                self.label_name.setObjectName("3")
                self.label_13.setObjectName("0")
                self.label_name.setObjectName("0")
                self.label_11.setText("厨余垃圾:%d" % self.y)
            elif self.label_13.objectName() == "4":
                self.label_13.setText("其余垃圾")
                self.m += 1
                self.label_name.setObjectName("4")
                self.label_13.setObjectName("0")
                self.label_name.setObjectName("0")
                self.label_12.setText("其余垃圾:%d" % self.m)

        self.label_13.objectNameChanged.connect(cao6)

        self.page_setting = QtWidgets.QWidget()
        self.page_setting.setObjectName("page_setting")
        self.label = QtWidgets.QLabel(self.page_setting)
        self.label.setGeometry(QtCore.QRect(0, 0, 900, 650))
        self.label.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, stop:0 rgba(0, 0, 0, 76), "
            "stop:1 rgba(255, 255, 255, 63)); font-size:100px;")
        self.label.setText("请传入视频或图片...")

        self.Page_Widget.addWidget(self.page_setting)
        self.label_Meau = QtWidgets.QLabel(self.main_widget)
        self.label_Meau.setGeometry(QtCore.QRect(10, 120, 161, 531))
        self.label_Meau.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0, y1:0.472, x2:1, y2:0.494, stop:0 rgba(0, 154, 162, "
            "105), stop:1 rgba(58, 104, 184, 204));")
        self.label_Meau.setText("")

        self.Btn_exit = QtWidgets.QPushButton(self.main_widget)
        self.Btn_exit.setGeometry(QtCore.QRect(10, 570, 161, 81))
        self.Btn_exit.setStyleSheet("QPushButton{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                    "font: 75 18pt \"Agency FB\";}\n"
                                    "QPushButton:hover{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                    "font: 75 18pt \"Agency FB\";\n"
                                    "}\n"
                                    "QPushButton:pressed{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                    "font: 75 18pt \"Agency FB\";\n"
                                    "}\n"
                                    "")
        self.Btn_meau = QtWidgets.QPushButton(self.main_widget)
        self.Btn_meau.setGeometry(QtCore.QRect(10, 120, 161, 81))
        self.Btn_meau.setStyleSheet("QPushButton{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                    "font: 75 18pt \"Agency FB\";}"
                                    "QPushButton:hover{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                    "font: 75 18pt \"Agency FB\";\n"
                                    "}\n"
                                    "QPushButton:pressed{\n"
                                    "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                    "font: 75 18pt \"Agency FB\";\n"
                                    "}\n"
                                    "")
        self.Btn_PageOne = QtWidgets.QPushButton(self.main_widget)
        self.Btn_PageOne.setGeometry(QtCore.QRect(10, 210, 161, 81))
        self.Btn_PageOne.setStyleSheet("QPushButton{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";}\n"
                                       "QPushButton:hover{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "QPushButton:pressed{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "")
        self.Btn_PageOne.setObjectName("Btn_PageOne")
        self.Btn_PageTwo = QtWidgets.QPushButton(self.main_widget)
        self.Btn_PageTwo.setGeometry(QtCore.QRect(10, 300, 161, 81))
        self.Btn_PageTwo.setStyleSheet("QPushButton{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";}\n"
                                       "QPushButton:hover{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "QPushButton:pressed{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "")
        self.Btn_PageThree = QtWidgets.QPushButton(self.main_widget)
        self.Btn_PageThree.setGeometry(QtCore.QRect(10, 390, 161, 81))
        self.Btn_PageThree.setStyleSheet("QPushButton{\n"
                                         "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                         "font: 75 18pt \"Agency FB\";}\n"
                                         "QPushButton:hover{\n"
                                         "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                         "font: 75 18pt \"Agency FB\";\n"
                                         "}\n"
                                         "QPushButton:pressed{\n"
                                         "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                         "font: 75 18pt \"Agency FB\";\n"
                                         "}\n"
                                         "")
        self.Btn_setting = QtWidgets.QPushButton(self.main_widget)
        self.Btn_setting.setGeometry(QtCore.QRect(10, 480, 161, 81))
        self.Btn_setting.setStyleSheet("QPushButton{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 241, 255, 164), stop:1 rgba(76, 173, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";}\n"
                                       "QPushButton:hover{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(184, 251, 255, 164), stop:1 rgba(144, 201, 243, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "QPushButton:pressed{\n"
                                       "background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:0.587, y2:0, stop:0 rgba(0, 154, 162, 164), stop:1 rgba(46, 104, 145, 213));\n"
                                       "font: 75 18pt \"Agency FB\";\n"
                                       "}\n"
                                       "")

        def cao7():
            if self.label_name.objectName() == "1":
                self.label_name.setText("1  有害垃圾" + "   " + str(self.w) + "   " + "ok")
                self.label_name.setObjectName("0")
            if self.label_name.objectName() == "2":
                self.label_name.setText("2  可回收垃圾" + "   " + str(self.h) + "   " + "ok")
                self.label_name.setObjectName("0")
            if self.label_name.objectName() == "3":
                self.label_name.setText("3  厨余垃圾" + "   " + str(self.y) + "   " + "ok")
                self.label_name.setObjectName("0")
            if self.label_name.objectName() == "4":
                self.label_name.setText("4  其余垃圾" + "   " + str(self.m) + "   " + "ok")
                self.label_name.setObjectName("0")

        self.label_name = QtWidgets.QLabel(self.main_widget)
        self.label_name.setGeometry(QtCore.QRect(20, 710, 871, 101))
        self.label_name.setStyleSheet("background-color: none;\n"
                                      "font: 26pt \"HGB4X_CNKI\";")
        self.label_name.setObjectName("0")
        self.label_name.objectNameChanged.connect(cao7)

        self.label_QQ = QtWidgets.QLabel(self.main_widget)
        self.label_QQ.setGeometry(QtCore.QRect(40, 30, 81, 81))

        self.label_name_bakground = QtWidgets.QLabel(self.main_widget)
        self.label_name_bakground.setGeometry(QtCore.QRect(0, 730, 781, 81))
        self.label_name_bakground.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0, y1:0.551, x2:1, y2:0.551, stop:0 rgba(0, 0, 0, 76), stop:1 rgba(255, 255, 255, 63));\n"
            "")
        self.label_name_bakground.setText("")
        MainWindow.setCentralWidget(self.main_widget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        def cao1():
            videoName, _ = QFileDialog.getOpenFileName(self, "打开", "", "*.avi;;*.mp4;;All Files(*)")
            # “”为用户取消
            if videoName != "":
                self.cap = cv2.VideoCapture(videoName)

                self.timer_camera.start(30)
                # 循环调用
                self.timer_camera.timeout.connect(self.openFrame)

        def cao2():
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开", "", "*.jpg;;*.png;;All Files(*)")
            if imgName != "":
                img = cv2.imread(imgName)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x, y, h = img.shape
                bytesPerLine = y * h
                frame = QImage(img.data, y, x, bytesPerLine, QImage.Format_RGB888). \
                    scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(frame))
                self.label.setObjectName("1")

        def cao3():
            if self.cap != []:
                self.cap.release()
                # 停止计时器
                self.timer_camera.stop()
                self.label.setText("视频被关闭了")
                self.label.setStyleSheet(
                    "background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, stop:0 rgba(0, 0, "
                    "0, 76), "
                    "stop:1 rgba(255, 255, 255, 63)); font-size:100px;")
            if self.label.objectName() == "1":
                self.label.setStyleSheet(
                    "background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, stop:0 rgba(0, 0, "
                    "0, 76), "
                    "stop:1 rgba(255, 255, 255, 63)); font-size:100px;")
                self.label.setText("图片被关闭")

        def cao4():
            self.w = 0
            self.label_2.setText("有害垃圾:%d" % self.w)
            self.h = 0
            self.label_7.setText("可回收垃圾:%d" % self.h)
            self.y = 0
            self.label_11.setText("厨余垃圾:%d" % self.y)
            self.m = 0
            self.label_12.setText("其余垃圾:%d" % self.m)
            self.label_13.setText("垃圾种类")
            self.label_13.setObjectName("0")
            self.label_name.setObjectName("0")
            self.label_name.setText("垃圾分类测试")

        self.retranslateUi(MainWindow)
        self.Page_Widget.setCurrentIndex(2)
        self.Btn_meau.clicked.connect(MainWindow.Meau)
        self.Btn_PageOne.clicked.connect(cao1)
        self.Btn_PageTwo.clicked.connect(cao2)
        self.Btn_PageThree.clicked.connect(cao3)
        self.Btn_setting.clicked.connect(cao4)
        self.Btn_exit.clicked.connect(MainWindow.Exit)
        self.Btn_exit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Btn_exit.setText(_translate("MainWindow", "退出"))
        self.Btn_meau.setText(_translate("MainWindow", "菜单"))
        self.Btn_PageOne.setText(_translate("MainWindow", "打开视频"))
        self.Btn_PageTwo.setText(_translate("MainWindow", "打开图片"))
        self.Btn_PageThree.setText(_translate("MainWindow", "关闭"))
        self.Btn_setting.setText(_translate("MainWindow", "重置"))
        self.label_name.setText(_translate("MainWindow", "垃圾分类测试"))


import Resource_rc

if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    ui = Designer()
    ui.show()
    sys.exit(app.exec_())