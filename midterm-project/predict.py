import pickle
from flask import Flask, request, jsonify
from sklearn.metrics import root_mean_squared_error

model_file = "model.bin"

with open(model_file, "rb") as file_in:
    dv, model = pickle.load(file_in)

app = Flask("energy_consumption")

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

@app.route("/predict", methods=["POST"])
def predict():
    home_energy = request.get_json()

    X = dv.transform([home_energy])
    y_pred = model.predict(X)

    y_test = [4.1]  # Replace with the actual true value for RMSE calculation

    global rmse
    rmse = rmse(y_test, y_pred)

    result = {
        "RMSE": rmse
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,  host="0.0.0.0", port=9696)