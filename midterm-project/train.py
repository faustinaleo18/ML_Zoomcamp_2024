import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Parameter
output_file = "model.bin"

# Data preparation
df = pd.read_csv("smart_home_energy_usage.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

categorical = ["occupancy_status", "appliance", "day_of_week", "season", "day_of_week"]
for cat in categorical:
    df[cat] = df[cat].str.lower().str.replace(' ', '_')

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["hour"] = df["timestamp"].dt.hour

day_of_week = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6
}

df.day_of_week = df.day_of_week.map(day_of_week)

df =  df[df["year"] != 2137]

del df["home_id"]

# Train Test Split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_valid = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train.energy_consumption_kwh.values
y_valid = df_valid.energy_consumption_kwh.values
y_test = df_test.energy_consumption_kwh.values
y_full_train = df_full_train.energy_consumption_kwh.values

df_train = df_train.drop(["energy_consumption_kwh", "timestamp"], axis=1)
df_valid = df_valid.drop(["energy_consumption_kwh", "timestamp"], axis=1)
df_test = df_test.drop(["energy_consumption_kwh", "timestamp"], axis=1)

df_full_train = df_full_train.drop(["energy_consumption_kwh", "timestamp"], axis=1)

# One Hot-Encoding
def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train.to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=15, random_state=1, n_jobs=-1, warm_start=True)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model): 
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict(X)

    return y_pred

# Training the final model
print ("In process of using the model...")

dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print(F"RMSE: {rmse} and R2 scores: {r2} from final model")

# Save the model
with open(output_file, "wb") as file_out:
    pickle.dump((dv, model), file_out)
print(f"Model is saved to {output_file}")