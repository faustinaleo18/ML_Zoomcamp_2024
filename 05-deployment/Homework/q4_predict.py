import pickle

from flask import Flask
from flask import request
from flask import jsonify

def load(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

dv_file = load("dv.bin")
model_file = load("model1.bin")

app = Flask("probability")

@app.route("/predict", methods=["POST"])
def predict():
    # request client information
    client = request.get_json()

    X = dv_file.transform([client])
    y_pred = model_file.predict_proba(X)[0,1]
    res = y_pred >= 0.5

    # return a result
    result = {
        "probability": float(y_pred),
        "result": bool(res)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)