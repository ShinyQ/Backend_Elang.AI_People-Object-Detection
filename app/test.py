from fastapi.testclient import TestClient
from main import app
import base64

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        'code': 200,
        'message': "Success",
        'data': "Object Detection API"
    }


def test_post_str_predict():
    with open("./image/street.jpg", "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.post(
        "/predict",
        json={'base64str': base64str}
    )

    assert response.status_code == 200


def test_post_image_predict():
    files = {'file': open('./image/street.jpg', 'rb')}

    response = client.post(
        "/predict_image",
        files=files
    )

    assert response.status_code == 200
