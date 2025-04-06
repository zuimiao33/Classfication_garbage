# 脚本运行依赖paddlex
# pip install paddlex

from distutils.log import error
from webbrowser import get
import paddlex as pdx
import cv2
import time




print("Loading model...")
# 模型
model = pdx.load_model('11')
print("Model loaded.")


# im = cv2.imread('3.jpg')
# im = im.astype('float32')

# 创建垃圾的字典
d = {'有害垃圾':0,
    '厨余垃圾':0,
    '可回收垃圾':0,
    '其他垃圾':0
    }

# 有害垃圾的数量（干电池（1 号、2 号、5 号））
class_youhai = ["Battery","Cigarette"]
num_1 = 0
# 厨余垃圾的数量（小土豆、切过的白萝卜、胡萝卜，尺寸为电池大小）
class_chuyu = ["Potato","White_radish","Carrot"]
num_2 = 0
# 可回收垃圾的数量（易拉罐、小号矿泉水瓶）
class_huishou = ["Cans","Bottle"]
num_3 = 0
# 其他垃圾的数量（瓷片、鹅卵石（小土豆大小））
class_qita = ["Ceramics","Pebbles"]
num_4 = 0


# 检测结果
dect = ""
# 置信度
rate = 0
# 检测是否成功的标志,0代表成功，1代表失败
flag_1 = 0


# flag 0 表示开始 flag 1 代表结束
flag = 0
import cv2
# 获取视频设备
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    cv2.imshow('video', frame)
    frame = frame.astype('float32')

    result = model.predict(frame)

    

    # 输出分类结果
    if model.model_type == "classifier":
        dect = result[0]['category']
        rate = result[0]['score']

        if dect in class_youhai and rate>=0.7:
            num_1 += 1
            d['有害垃圾']=num_1
            print("1 "+"有害垃圾 ",d.get('有害垃圾')," OK!")
            flag = 0

        elif dect in class_chuyu and rate>=0.7:
            num_2 += 1
            d['厨余垃圾']=num_2
            print("2 "+"厨余垃圾 ",d.get('厨余垃圾')," OK!")
            flag = 0


        elif dect in class_huishou and rate>=0.7:
            num_3 += 1
            d['可回收垃圾']=num_3
            print("3 "+"可回收垃圾 ",d.get('可回收垃圾')," OK!")
            flag = 0

        elif rate>=0.7:
            num_4 += 1
            d['其他垃圾']=num_4
            print("4 "+"其他垃圾 ",d.get('其他垃圾')," OK!")
            flag = 0

        
        else:
            # print("Error!")
            flag = 1

        if flag == 0:
            print(result)
            time.sleep(1)
        
        if flag == 1:
            time.sleep(0.1)
            


    # 等待键盘事件，如果为q，退出
    key = cv2.waitKey(10)
    if key & 0xff == ord('q'):
        break

    # 等待键盘事件，如果为k,重置计数
    key = cv2.waitKey(10)
    if key & 0xff == ord('k'):
        num_1 = 0
        num_2 = 0
        num_3 = 0
        num_4 = 0


# 释放videoCapture
cap.release()
cv2.destroyAllWindows()
