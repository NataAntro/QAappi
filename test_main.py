from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_question_answering():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"question": "Что такое API?"}
    
def test_example():
    model_name = "AndrewChar/model-QA-5-epoch-RU"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
    'question': 'Что такое API?',
    'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
}
    res = nlp(QA_input)
    response = client.post("/predict/", json={"answer": res.get('answer')})
    assert response.status_code == 200
    assert response.json() == {"answer": res.get('answer')}
   
    
