import pickle

def load(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

dv_file = load("dv.bin")
model_file = load("model1.bin")

client = {
    "job": "management", 
    "duration": 400, 
    "poutcome": "success"
}

X = dv_file.transform([client])
y_pred = model_file.predict_proba(X)[0,1]

print(y_pred)