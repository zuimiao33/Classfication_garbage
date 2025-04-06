from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QPropertyAnimation, QRect

from main import Designer

global Temp_2
Temp_2 = 0
class UiFunction(Designer):
    def MeauFunction(self):
        # print(1)
        global Temp_2
        if Temp_2 == 0:
            Temp_2 = 1
            animation = QPropertyAnimation(self.label_Meau)
            animation.setTargetObject(self.label_Meau)  # 设置动画对象
            animation.setPropertyName(b"geometry")
            animation.setStartValue(QRect(self.label_Meau.geometry().x(),
                                          self.label_Meau.geometry().y(), 161, 561))  # 设置起始点;初始尺寸
            animation.setEndValue((QRect(self.label_Meau.geometry().x(),
                                         self.label_Meau.geometry().y(), 161, 101)))  # 设置终点；终止尺寸
            animation.setDuration(200)  # 时长单位毫秒
            animation.start()
            self.Btn_PageOne.hide()
            self.Btn_PageTwo.hide()
            self.Btn_PageThree.hide()
            self.Btn_setting.hide()
            self.Btn_exit.hide()
        else:
            Temp_2 = 0
            animation = QPropertyAnimation(self.label_Meau)
            animation.setTargetObject(self.label_Meau)  # 设置动画对象
            animation.setPropertyName(b"geometry")
            animation.setStartValue(QRect(self.label_Meau.geometry().x(),
                                          self.label_Meau.geometry().y(), 161, 101))  # 设置起始点;初始尺寸
            animation.setEndValue((QRect(self.label_Meau.geometry().x(),
                                         self.label_Meau.geometry().y(), 161, 561)))  # 设置终点；终止尺寸
            animation.setDuration(200)  # 时长单位毫秒
            animation.start()
            self.Btn_PageOne.show()
            self.Btn_PageTwo.show()
            self.Btn_PageThree.show()
            self.Btn_setting.show()
            self.Btn_exit.show()
    def Shaow(self):
        self.label_QQ.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=40, xOffset=4, yOffset=4,
                                                                            color=QtGui.QColor(73, 229,
                                                                                               237)))
        self.label_Meau.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=40, xOffset=4, yOffset=4,
                                                                              color=QtGui.QColor(73, 229,
                                                                                                 237)))
        self.label_name_bakground.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(blurRadius=40, xOffset=4, yOffset=4,
                                                color=QtGui.QColor(77, 200,
                                                                   255)))