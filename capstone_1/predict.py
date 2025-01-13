import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

model_interpreter = tflite.Interpreter(model_path="rice-leaf-disease-prediction-model.tflite")
model_interpreter.allocate_tensors()

input_index = model_interpreter.get_input_details()[0]['index']
output_index = model_interpreter.get_output_details()[0]['index']

classes = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).resize((260, 260))  # Resize image
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array / 255.0  # Normalize image array

def predict(url):
    try:
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception("Failed to retrieve image from URL")

        image_bytes = response.content  # Get image content in bytes
        input_data = preprocess_image(image_bytes)

        model_interpreter.set_tensor(input_index, input_data)
        model_interpreter.invoke()
        predictions = model_interpreter.get_tensor(output_index)[0]

        float_predictions = predictions.tolist()
        return dict(zip(classes, float_predictions))

    except Exception as e:
        return {"error": str(e)}

@app.route('/2015-03-31/functions/function/invocations', methods=['POST'])
def flask_predict():
    data = request.get_json() 

    if 'url' not in data:
        return jsonify({"error": "No image URL provided"}), 400  

    url = data['url']
    result = predict(url)

    return jsonify(result)  

# Run the Flask app locally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  