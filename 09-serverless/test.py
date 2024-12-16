import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

data = {"image": "yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}

result = requests.post(url, json=data).json()
print(result)