from fastapi import FastAPI, Response
from torchvision import transforms
from pydantic import BaseModel
from .helper import api
from .model import object

app = FastAPI()


class Input(BaseModel):
    base64str: str
    threshold: float


@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Object Detection API", response.status_code)


@app.post("/predict", status_code=200)
def get_prediction(d: Input, response: Response):
    model, COCO_INSTANCE_CATEGORY_NAMES = object.initialize()

    img = object.base64str_to_PILImage(d.base64str)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [
        [{'x1': float(i[0]), 'y1': float(i[1])}, {'x2': float(i[2]), 'y2': float(i[3])}]
        for i in list(pred[0]['boxes'].detach().numpy())
    ]

    pred_score = list(pred[0]['scores'].detach().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > d.threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]

    return api.builder({'boxes': pred_boxes, 'classes': pred_class}, response.status_code)
