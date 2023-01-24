from fastapi import FastAPI
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

app = FastAPI()
model_name = "AndrewChar/model-QA-5-epoch-RU"

@app.get("/predict")
def predict():
    """Анализ текста по контексту в формате вопрос-ответ"""
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Что такое API?',
        'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
    }
    res = nlp(QA_input)
    return {"answer": res.get('answer')}
