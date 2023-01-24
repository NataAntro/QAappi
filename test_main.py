from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_question_answering():
    response = client.get("/predict/")
    assert response.status_code == 200
    response_json = response.json()
    assert "question" in response_json
    assert response_json["question"] == "Что такое API?"
    assert "context" in response_json
    assert response_json["context"] == "API — описание способов взаимодействия одной компьютерной программы с другими."
    assert "answer" in response_json
    assert response_json['answer'] == 'описание способов взаимодействия одной компьютерной программы'
    
