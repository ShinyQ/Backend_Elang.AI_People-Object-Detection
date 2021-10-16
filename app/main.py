from fastapi import FastAPI, Response, UploadFile, File
from pydantic import BaseModel
from helper import api
from model import object
from PIL import Image
import numpy as np

app = FastAPI()


class Input(BaseModel):
    base64str: str


@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Object Detection API", response.status_code)


@app.post("/predict", status_code=200)
def get_prediction(d: Input, response: Response):
    img = object.base64str_to_PILImage(d.base64str)
    pred_boxes, pred_class = object.model_prediction(img)

    return api.builder({'boxes': pred_boxes, 'classes': pred_class}, response.status_code)


@app.post("/predict_image", status_code=200)
def get_prediction_image(file: UploadFile = File(...)):
    img = Image.open(file.file)
    pred_boxes, pred_class = object.model_prediction(img)
    print(pred_class)
    del_arr = []

    for i, val in enumerate(pred_class):
        if val != "person":
            del_arr.append(i)

    for index in sorted(del_arr, reverse=True):
        del pred_boxes[index]
        del pred_class[index]

    return api.builder({'boxes': pred_boxes, 'classes': pred_class}, 200)
