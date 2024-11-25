import requests

url = "http://localhost:9696/predict"

home = {
    "timestamp": "2023-01-01 00:00:00",
    "energy_consumption_kwh": 2.87,
    "temperature_setting_c": 22.1,
    "occupancy_status": "occupied",
    "appliance": "refrigerator",
    "usage_duration_minutes": 111,
    "season": "spring",
    "day_of_week": "sunday",
    "holiday": 0}

response = requests.post(url, json=home).json()
print(response)