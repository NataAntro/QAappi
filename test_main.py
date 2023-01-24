from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/predict/")
    QA_input = {
        'question': 'Что такое API?',
        'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
    }
    json_data = response.json() 
                          
    assert response.status_code == 200
    assert json_data['answer'] == 'описание способов взаимодействия одной компьютерной программы'
    
