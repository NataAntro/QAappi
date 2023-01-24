from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_question_answering():
    response = client.get("/predict/")
    response_json = response.json()
    assert "question" in response_json
    assert response_json["question"] == "Что такое API?"
    assert "context" in response_json
    assert response_json["context"] == "API — описание способов взаимодействия одной компьютерной программы с другими."
    assert "answer" in response_json
    assert response_json["answer"] == "описание способов взаимодействия одной компьютерной программы"
    
def test_example():
    model_name = "AndrewChar/model-QA-5-epoch-RU"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
    'question': 'Что такое API?',
    'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
}
    res = nlp(QA_input)
    response = client.post("/predict/", json={"answer": res.get('answer')})
    assert response.json() == {"answer": res.get('answer')}
   
    
