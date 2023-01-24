
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_qa():
    response = client.get("/predict/")
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Что такое API?',
        'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'}
    
    json_data = response.json() 
                          
    assert response.status_code == 200
    assert json_data['answer'] == 'описание способов взаимодействия одной компьютерной программы'
    
