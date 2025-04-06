import json
import os
from time import sleep

import cv2
import numpy as np
import onnxruntime as rt
from PIL import Image

EXPORT_MODEL_VERSION = 1


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

    def process_output(self, fetches: dict, outputs: dict) -> dict:
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
        return result


def main():
    dir_path = os.getcwd()
    model = ONNXModel(dir_path=dir_path)
    model.load()
    cap = cv2.VideoCapture(0    )
    while True:
        ret, frame = cap.read()
        cv2.imshow('windowName', frame)
        # frame = cv2.imread(r"F:\USUALLY\rubbish ONNX\example\1.jpg")
        img = Image.fromarray(frame, "RGB")

        # image = Image.open(img)
        outputs = model.predict(img)
        print(f"Predicted: {outputs}")
        # 点击小写字母q 退出程序
        if cv2.waitKey(1) == ord('q'):
            break

        # 点击窗口关闭按钮退出程序
        if cv2.getWindowProperty('windowName', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
