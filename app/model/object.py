import torchvision
import base64
import io

from torchvision import transforms
from PIL import Image
from torch import nn


def initialize():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person']
    model.roi_heads.box_predictor.cls_score = nn.Linear(1024, len(COCO_INSTANCE_CATEGORY_NAMES))

    return model.eval(), COCO_INSTANCE_CATEGORY_NAMES


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)

    return Image.open(bytesObj)


def model_prediction(img):
    model, COCO_INSTANCE_CATEGORY_NAMES = initialize()
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]

    pred_boxes = [
        [{'x1': float(i[0]), 'y1': float(i[1])}, {'x2': float(i[2]), 'y2': float(i[3])}]
        for i in list(pred[0]['boxes'].detach().numpy())
    ]

    pred_score = list(pred[0]['scores'].detach().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > 0][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]

    return pred_boxes, pred_class
