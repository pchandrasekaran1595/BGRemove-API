from typing import Union

from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from static.utils import models, decode_image, encode_image_to_base64, preprocess_replace_bg_image


VERSION: str = "0.0.1"
STATIC_PATH: str = "static"


class APIData(BaseModel):
    imageData_1: str
    imageData_2: Union[str, None] = None


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


@app.get("/{infer_type}")
async def get_infer(infer_type: str):
    if infer_type == "remove":
        return JSONResponse({
            "statusText" : "Background Removal Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif infer_type == "replace":
        return JSONResponse({
            "statusText" : "Background Replacement Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    else:
        return JSONResponse({
            "statusText" : "Invalid Infer Type",
            "statusCode" : status.HTTP_400_BAD_REQUEST,
            "version" : VERSION,
        })
    


@app.get("/{infer_type}/li")
async def get_remove_bg(infer_type: str):
    if infer_type == "remove":
        return JSONResponse({
            "statusText" : "Background Removal Endpoint (Lightweight Model)",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })

    elif infer_type == "replace":
        return JSONResponse({
            "statusText" : "Background Replacement Endpoint (Lightweight Model)",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    else:
        return JSONResponse({
            "statusText" : "Invalid Infer Type",
            "statusCode" : status.HTTP_400_BAD_REQUEST,
            "version" : VERSION,
        })


@app.post("/{infer_type}")
async def get_remove_bg(infer_type: str, images: APIData):

    if infer_type == "remove":
        _, image = decode_image(images.imageData_1)

        mask = models[0].infer(image=image)
        for i in range(3): image[:, :, i] = image[:, :, i] & mask

        return JSONResponse({
            "statusText" : "Background Removal Complete",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
            "maskImageData" : encode_image_to_base64(image=mask),
            "bglessImageData" : encode_image_to_base64(image=image),
        })
    
    elif infer_type == "replace":
        _, image_1 = decode_image(images.imageData_1)
        _, image_2 = decode_image(images.imageData_2)

        mask = models[0].infer(image=image_1)
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
    
    else:
        return JSONResponse({
            "statusText" : "Invalid Infer Type",
            "statusCode" : status.HTTP_400_BAD_REQUEST,
            "version" : VERSION,
        })

@app.post("/{infer_type}/li")
async def get_remove_bg(infer_type: str, images: APIData):
    
    if infer_type == "remove":
        _, image = decode_image(image.imageData)

        mask = models[1].infer(image=image)
        for i in range(3): image[:, :, i] = image[:, :, i] & mask

        return JSONResponse({
            "statusText" : "Background Removal Complete",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
            "maskImageData" : encode_image_to_base64(image=mask),
            "bglessImageData" : encode_image_to_base64(image=image),
        })
    
    elif infer_type == "replace":
        _, image_1 = decode_image(images.imageData_1)
        _, image_2 = decode_image(images.imageData_2)

        mask = models[1].infer(image=image_1)
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

    else:
        return JSONResponse({
            "statusText" : "Invalid Infer Type",
            "statusCode" : status.HTTP_400_BAD_REQUEST,
            "version" : VERSION,
        })