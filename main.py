import cv2
from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from static.utils import CFG, decode_image, encode_image_to_base64, preprocess_replace_bg_image


cfg_full = CFG()
cfg_full.setup()
cfg_light = CFG(lightweight=True)
cfg_light.setup()

VERSION: str = "0.0.1"
STATIC_PATH: str = "static"


class Remove(BaseModel):
    imageData: str


class Replace(BaseModel):
    imageData_1: str
    imageData_2: str


origins = [
    "http://localhost:3031",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint of Background Removal/Replacement API",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/version")
async def version():
    return JSONResponse({
        "statusText" : "Background Removal/Replacement API Version Fetch Successful",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/remove")
async def get_remove_bg():
    return JSONResponse({
        "statusText" : "Background Removal Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/replace")
async def get_replace_bg():
    return JSONResponse({
        "statusText" : "Background Replacement Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/remove/li")
async def get_remove_bg():
    return JSONResponse({
        "statusText" : "Background Removal Endpoint (Lightweight Model)",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/replace/li")
async def get_replace_bg():
    return JSONResponse({
        "statusText" : "Background Replacement Endpoint (Lightweight Model)",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/remove")
async def get_remove_bg(image: Remove):
    _, image = decode_image(image.imageData)

    mask = cfg_full.infer(image=image)
    for i in range(3): image[:, :, i] = image[:, :, i] & mask

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "maskImageData" : encode_image_to_base64(image=mask),
        "bglessImageData" : encode_image_to_base64(image=image),
    })


@app.post("/replace")
async def get_replace_bg(images: Replace):
    _, image_1 = decode_image(images.imageData_1)
    _, image_2 = decode_image(images.imageData_2)

    mask = cfg_full.infer(image=image_1)
    mh, mw = mask.shape
    image_2 = preprocess_replace_bg_image(image_2, mw, mh)
    for i in range(3): 
        image_1[:, :, i] = image_1[:, :, i] & mask
        image_2[:, :, i] = image_2[:, :, i] & (255 - mask) 

    image_2 += image_1   

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "bgreplaceImageData" : encode_image_to_base64(image=image_2),
    })


@app.post("/remove/li")
async def get_remove_bg(image: Remove):
    _, image = decode_image(image.imageData)

    mask = cfg_light.infer(image=image)
    for i in range(3): image[:, :, i] = image[:, :, i] & mask

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "maskImageData" : encode_image_to_base64(image=mask),
        "bglessImageData" : encode_image_to_base64(image=image),
    })


@app.post("/replace/li")
async def get_replace_bg(images: Replace):
    _, image_1 = decode_image(images.imageData_1)
    _, image_2 = decode_image(images.imageData_2)

    mask = cfg_light.infer(image=image_1)
    mh, mw = mask.shape
    image_2 = preprocess_replace_bg_image(image_2, mw, mh)
    for i in range(3): 
        image_1[:, :, i] = image_1[:, :, i] & mask
        image_2[:, :, i] = image_2[:, :, i] & (255 - mask) 

    image_2 += image_1   

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "bgreplaceImageData" : encode_image_to_base64(image=image_2),
    })