import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {"url": "https://bit.ly/leaf-capstone-1"}

response = requests.post(url, json=data)

print(response.json())