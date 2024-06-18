from typing import Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # Optional for cross-origin requests
import cv2
import numpy as np
from MedicineDetails import RecognizeMedicineInfo
from paddleocr import PaddleOCR

app = FastAPI()

# Optional: Enable CORS if you need to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins, or ["http://localhost", ...] for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def RecognizeMedicineInfo(image):
    
    # imagepath = r"D:\Graduation Project\RealVersionOfModel version Six (Last Update)\TestImages\All-Vent\huawei p30 418.jpg"
    # image = cv2.imread(image_path)

    pipline = PaddleOCR(rec_algorithm='CRNN',use_angle_cls=True)


    ocr = pipline
    result = ocr.ocr(image, cls=True)
    words = [row[1][0] for row in result[0]]
    return words


# Endpoint to accept image file and return its height and width
@app.post("/get_image_dimensions/")
async def get_image_dimensions(file: UploadFile = File(...)):
    # Read image file as bytes
    image_bytes = await file.read()
    
    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode numpy array to image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Get image dimensions (height and width)
    height, width, _ = image.shape
    
    
    
    pipline = PaddleOCR(rec_algorithm='CRNN',use_angle_cls=True)


    ocr = pipline
    result = ocr.ocr(image, cls=True)
    words = [row[1][0] for row in result[0]]
    
    return words

# Optional: Include a root endpoint for documentation
@app.get("/")
async def root():
    return {"message": "Welcome to the Image Dimensions API!"}
