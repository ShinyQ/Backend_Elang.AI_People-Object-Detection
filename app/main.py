from fastapi import FastAPI, Response
from helper import api

app = FastAPI()


@app.get('/', status_code=200)
def index(response: Response):
    return api.builder("Hello World", response.status_code)
