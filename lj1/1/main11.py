# from re import S
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
import time
from AnimFunction import *

import json
import os
from time import sleep
import numpy as np
import onnxruntime as rt
from PIL import Image

EXPORT_MODEL_VERSION = 1
youhai = ["Battery"]
huishou = ["Bottle","Cans"]
chuyu = ["Vegetables"]
qita = ["Ceramics"]


class ONNXModel:
    def __init__(self, dir_path) -> None:
    
        """获取模型文件名的方法"""
        model_dir = dir_path
        # 打开signature.json
        with open(os.path.join(model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(model_dir, self.signature.get("filename"))
        # 判断模型文件是否存在
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        # 获取模型输入和输出的签名
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")
        self.session = None
        if "Image" not in self.signature_inputs:
            raise ValueError(
                "ONNX model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")
        # 在签名文件中查找版本。
        # 如果未找到或与预期不匹配，打印消息
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def load(self) -> None:
        """将模型从路径加载到模型文件"""
        # 将 ONNX 模型加载为session.
        self.session = rt.InferenceSession(path_or_bytes=self.model_file)

    def predict(self, image: Image.Image):
        """
        用ONNX模型预测session!
        """
        # 处理图像以与模型兼容
        img = self.process_image(image, self.signature_inputs.get("Image").get("shape"))
        # run the model!
        fetches = [(key, value.get("name")) for key, value in self.signature_outputs.items()]
        # make the image a batch of 1
        feed = {self.signature_inputs.get("Image").get("name"): [img]}
        outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
        return self.process_output(fetches, outputs)

    def process_image(self, image: Image.Image, input_shape: list) -> np.ndarray:
        """
        给定 PIL 图像，将正方形中心裁剪并调整大小以适合预期的模型输入，并从 [0，255] 转换为 [0，1] 值。
        """
        width, height = image.size
        # 确保图像类型与模型兼容，如果不匹配，则进行转换
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 居中裁剪图像（可以替换任何其他方法来制作方形图像，例如仅调整大小或用 0 填充边缘）
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # 裁剪图像的中心
            image = image.crop((left, top, right, bottom))
        # 现在图像是正方形的，将其大小调整为模型输入的正确形状
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # 使 0-1 浮点数而不是 0-255 int（默认情况下加载 PIL 图像）
        image = np.asarray(image) / 255.0
        # 按照模型预期设置输入格式
        return image.astype(np.float32)

    def process_output(self, fetches: dict, outputs: dict) :
        # 取消批处理，因为我们运行了一个批量大小为 1 的图像，
        # 使用 tolist（） 转换为普通的 python 类型，并使用 .decode（） 将任何字节字符串转换为普通字符串
        out_keys = ["label", "confidence"]
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        # 取出概率最大的值
        max_confs = max(confs)
        # 取出概率最大值对应的索引
        id_max = confs.index(max_confs)
        result = labels[id_max] + " : " + str(max_confs)
        return labels[id_max],max_confs,result




class model(QtCore.QThread):
    _display = pyqtSignal(int)

    def __init__(self):
        super().__init__()
    
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.count4 = 0
        self.x = 0
        self.w = None

    def classify(self, labels):
        if labels in youhai:
            if self.count1>30:
                self.x = 1
                self._display.emit(self.x)
                self.count1 = 0
            else:
                self.count1+=1
                self.count2=self.count3=self.count4 = 0

        elif labels in huishou:
            if self.count2>30:
                self.x = 2
                self._display.emit(self.x)
                self.count2 = 0
            else:
                self.count2+=1
                self.count1=self.count3=self.count4 = 0 

        elif labels in chuyu:
            if self.count3>30:
                self.x = 3
                self._display.emit(self.x)
                self.count3 = 0
            else:
                self.count3+=1
                self.count1=self.count2=self.count4 = 0 

        else:
            if self.count4>30:
                self.x = 4
                self._display.emit(self.x)
                self.count4 = 0
            else:
                self.count4+=1
                self.count1=self.count2=self.count3 = 0        


    def run(self):
        dir_path = os.getcwd()
        model = ONNXModel(dir_path=dir_path)
        model.load()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('windowName', frame)
            # frame = cv2.imread(r"F:\USUALLY\rubbish ONNX\example\1.jpg")
            img = Image.fromarray(frame, "RGB")

            # image = Image.open(img0)
            outputs,conf,res = model.predict(img)
            if conf>=0.9:
                self.w = outputs
                self.classify(outputs)
            else:
                pass
            # 点击小写字母q 退出程序
            if cv2.waitKey(1) == ord('q'):
                break

            # 点击窗口关闭按钮退出程序
            if cv2.getWindowProperty('windowName', cv2.WND_PROP_AUTOSIZE) < 1:
                break

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
        # self.model = pdx.load_model('11')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setupUi(self)
        #       光影效果
        UiFunction.Shaow(self)
        self.thread1 = model()
        self.thread1._display.connect(self.updateobjname)
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

    # def showimg(self, img):
    #     img = img.scaled(self.label_16.width(), self.label_16.height())
    #     self.label_16.setPixmap(QPixmap.fromImage(img))

    def setupUi(self, MainWindow):
        MainWindow.resize(2400, 1600)
        desktop = QApplication.desktop()
        rect = desktop.frameSize()
        MainWindow.resize(QSize(rect.width(), rect.height()))

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
        self.label_2.setGeometry(QtCore.QRect(850, 50, 360, 100))
        self.label_2.setStyleSheet("background-color:None;\n"
                                   "border-radius:20px; font-size:50px;")
        self.w = 0
        self.label_2.setText("有害垃圾: %d" % self.w)

        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(850, 180, 360, 100))
        self.label_7.setStyleSheet("background-color:None;\n"
                                   "border-radius:20px; font-size:50px;")
        self.label_7.setText("可回收垃圾: %d" % self.h)

        self.label_11 = QtWidgets.QLabel(self)
        self.label_11.setGeometry(QtCore.QRect(850, 310, 360, 100))
        self.label_11.setStyleSheet("background-color:None;\n"
                                    "border-radius:20px;font-size:50px;")
        self.label_11.setText("厨余垃圾: %d" % self.y)

        self.label_12 = QtWidgets.QLabel(self)
        self.label_12.setGeometry(QtCore.QRect(850, 440, 360, 100))
        self.label_12.setStyleSheet("background-color:None;\n"
                                    "border-radius:20px;font-size:50px;")
        self.label_12.setText("其余垃圾: %d" % self.m)

        self.label_13 = QtWidgets.QLabel(self)
        self.label_13.setText(" 垃圾种类")
        self.label_13.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, "
                                    "stop:0 rgba(0, 0, "
                                    "0, 76), "
                                    "stop:1 rgba(255, 255, 255, 63)); font-size:75px;border-radius:30px ")
        self.label_13.resize(400, 480)
        self.label_13.move(1150, 500)
        self.label_13.setObjectName("0")
        self.n = self.label_13.objectName()

        # self.label_16 = QtWidgets.QLabel(self)
        # self.label_16.resize(800, 500)
        # self.label_16.move(1100, 550)
        # self.label_16.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, "
        #                             "stop:0 rgba(0, 0, "
        #                             "0, 76), "
        #                             "stop:1 rgba(255, 255, 255, 63)); font-size:75px;border-radius:30px ")

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
        self.label.setGeometry(QtCore.QRect(0, 0, 630, 455))
        self.label.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0.497, y1:0, x2:0.493, y2:1, stop:0 rgba(0, 0, 0, 76), "
            "stop:1 rgba(255, 255, 255, 63)); font-size:60px;")
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
            self.label_2.setText("有害垃圾: %d" % self.w)
            self.h = 0
            self.label_7.setText("可回收垃圾: %d" % self.h)
            self.y = 0
            self.label_11.setText("厨余垃圾: %d" % self.y)
            self.m = 0
            self.label_12.setText("其余垃圾: %d" % self.m)
            self.label_13.setText(" 垃圾种类")
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