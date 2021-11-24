from fastapi import FastAPI, File, UploadFile
# To run a FastAPI application in a remote server machine is an ASGI server program like Uvicorn.
import uvicorn
import numpy as np
import tensorflow as tf
from image_classification import classify_image
from fastapi.middleware.cors import CORSMiddleware

app= FastAPI()

# To allow Cross Origin Requests
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def welcome_msg():
    return "Hello World"


@app.get("/user")
async def welcome_msg(name: str):
    return f"Hello {name}"    


@app.post("/classify")
async def predict(file: UploadFile = File(...)):
    # classify image into one of the 6 classes
    classification_results = classify_image(await file.read())
    # attach uploaded filename in result
    #classification_results["uploaded_filename"] = file.filename 
    return classification_results


if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8001)    
