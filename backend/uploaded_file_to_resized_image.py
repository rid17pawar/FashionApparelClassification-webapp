from io import BytesIO
from PIL import Image
import numpy as np

# Constants
IMAGE_SIZE = 60

def convert_uploaded_file_to_image(data):
    pil_image = Image.open(BytesIO(data))
    img_batch = resize_image(pil_image)
    return img_batch

def resize_image(image: Image.Image):
    image = image.resize([IMAGE_SIZE, IMAGE_SIZE])
    #shapeimg= np.shape(image)
    
    image = np.asfarray(image)
    image = image/255.0
    # predict() takes batch of images as input not a single image so, add 1 dimension
    img_batch = np.expand_dims(image, 0) 
    #shapeimgbatch = np.shape(img_batch)
    return img_batch    
