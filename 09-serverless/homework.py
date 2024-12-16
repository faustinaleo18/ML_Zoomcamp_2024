import numpy as np
import os
from PIL import Image
import tensorflow.lite as tflite
import json

MODEL_NAME = os.getenv("MODEL_NAME", "model_2024_hairstyle.tflite")

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# image = "yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

def predict(image_path):
    image = Image.open(image_path)
    image = prepare_image(image, target_size=(200, 200))

    image = np.array(image, dtype="float32")
    image = np.array([image])
    image = prepare_input(image)

    interpreter.set_tensor(input_index, image)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])

def lambda_handler(event, context):
    image_path = event["image"]
    pred = predict(image_path)
    result = {
        "prediction": pred
    }
    return json.dumps(result)