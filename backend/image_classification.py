# data can be kept as bytes in an in-memory buffer when we use the io module's Byte IO operations.
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from uploaded_file_to_resized_image import convert_uploaded_file_to_image

# Constants
MODEL = tf.keras.models.load_model("models/cnn_model11.h5")
CLASSES = ['Bottomwear', 'Kurti', 'Loungewear and Nightwear', 'One Piece Dress', 'Saree', 'Topwear']
TOTAL_CLASSES = 6
IMAGE_SIZE = 60

def classify_image(uploaded_file):
    # convert uploaded file into bytes and then into numpy array
    image_batch = convert_uploaded_file_to_image(uploaded_file)

    predictions = MODEL.predict(image_batch) 
    predict_probability = np.amax(predictions)*100
    product= CLASSES[np.argmax(predictions)]

    # for debugging purpose: display top-3 results which are > 10% probability
    """ top_predictions = {}
    if(predict_probability < 90):
        count=0
        for i in range(0,TOTAL_CLASSES):
            if((predictions[0][i]*100 > 9) and (count<3)):
                count = count + 1
                top_predictions[CLASSES[i]] = float(str(round(predictions[0][i]*100, 2)))
        top_predictions = sorted(top_predictions.items(), key= lambda kv: kv[1], reverse=True)
    else:
        top_predictions[product] = round(predict_probability, 2) """

    # for debugging purpose: display all probabilities
    """ all_probabilities= {}
    for i in range(0,TOTAL_CLASSES):
        temp = predictions[0][i]*100
        temp = str(round(temp, 2)) + "%"
        all_probabilities[CLASSES[i]] = temp """

    # for debugging purpose:
    """ return {
        "prediction_probabilities" : all_probabilities,
        "predicted_subcategory" : product,
        "confidence" : str(round(predict_probability, 2))+"%",
        "Top_predictions" : top_predictions
        }     """

    return {
        "predicted_class" : product,
        "confidence" : str(round(predict_probability, 2))+"%",
    }

    