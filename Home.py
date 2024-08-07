import requests

url = "http://127.0.0.1:5000/ask"
response = requests.get(url)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to get a response:", response.status_code)