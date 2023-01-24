from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_question_answering():
    response = client.get("/predict/",
        json= {
    "question": "Что такое API?",
    "context": "API — описание способов взаимодействия одной компьютерной программы с другими."
}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert response_json['answer'] == 'описание способов взаимодействия одной компьютерной программы'
    
