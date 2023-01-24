from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert res.get() == { "context": "API — описание способов взаимодействия одной компьютерной программы с другими.",
    "question": "Что такое API?"
}

def test_question_answering():
    response = client.post("/predict/",
        json= { "context": "API — описание способов взаимодействия одной компьютерной программы с другими.",
    "question": "Что такое API?"
}
    )
    json_data = response.json() 

    assert response_json['answer'] == 'описание способов взаимодействия одной компьютерной программы'
    
