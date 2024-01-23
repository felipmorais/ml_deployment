import requests
import json

passageiro = {
    "Pclass": "1st",
    "Sex": "Male",
    "Age": 38,
    "SibSp": 1,
    "Parch": 1,
    "Fare": 72.4,
    "Embarked": "Cherbourg",
    # 'Survived': 1,
    # 'CorrectPrediction': True,
}
json_passageiro = json.dumps(passageiro)
response = requests.post("http://localhost:8000/predict", json=json_passageiro)
print(response)
if response.status_code == 200:
    response_data = response.json()
    print(f"Response: {response_data}")
else:
    print("Error fetching the data")