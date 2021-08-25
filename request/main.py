import base64
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import requests

from PIL import Image

with open("../app/image/street3.jpg", "rb") as image_file:
    base64str = base64.b64encode(image_file.read()).decode("utf-8")

payload = json.dumps({"base64str": base64str})

response = requests.post("http://127.0.0.1:8000/predict", data=payload)
data_dict = response.json()

print(data_dict)


def PILImage_to_cv2(img):
    return np.asarray(img)


def drawboundingbox(img, boxes, pred_cls, rect_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    for cat in pred_cls:
        class_color_dict[cat] = [255]

    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0]['x1']), int(boxes[i][0]['y1'])),
            (int(boxes[i][1]['x2']), int(boxes[i][1]['y2'])),
            color=class_color_dict[pred_cls[i]], thickness=rect_th
        )

    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


img = Image.open("../app/image/street3.jpg")
drawboundingbox(img, data_dict['data']['boxes'], data_dict['data']['classes'])
