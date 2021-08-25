import torchvision
import base64
import io
from PIL import Image
from torch import nn


def initialize():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person'
    ]

    model.roi_heads.box_predictor.cls_score = nn.Linear(1024, len(COCO_INSTANCE_CATEGORY_NAMES))

    return model.eval(), COCO_INSTANCE_CATEGORY_NAMES


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img
