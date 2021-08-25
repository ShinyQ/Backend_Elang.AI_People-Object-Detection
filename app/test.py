from fastapi.testclient import TestClient
from main import app
import base64

client = TestClient(app)

predict_response = {
    "code": 200,
    "message": "Success",
    "data": {
        "boxes": [
            [
                {
                    "x1": 13.274473190307617,
                    "y1": 12.517145156860352
                },
                {
                    "x2": 712.6762084960938,
                    "y2": 658
                }
            ],
            [
                {
                    "x1": 15.296076774597168,
                    "y1": 0
                },
                {
                    "x2": 670.1613159179688,
                    "y2": 648.7072143554688
                }
            ]
        ],
        "classes": [
            "cat",
            "dog"
        ]
    }
}


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        'code': 200,
        'message': "Success",
        'data': "Object Detection API"
    }


def test_post_str_predict():
    with open("./image/dog_cat.JPG", "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.post(
        "/predict",
        json={'base64str': base64str}
    )

    assert response.status_code == 200
    assert response.json() == predict_response


def test_post_image_predict():
    files = {'file': open('./image/dog_cat.JPG', 'rb')}

    response = client.post(
        "/predict_image",
        files=files
    )

    assert response.status_code == 200
    assert response.json() == predict_response
