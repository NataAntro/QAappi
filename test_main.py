from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.testclient import TestClient
from FastAPI import app

client = TestClient(app)

def test_question_answering(model_name: str, question: str, context: str):
    # Инициализация конвейера вопрос-ответ
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    # Задание входных данных для конвейера
    QA_input = {'question': question, 'context': context}
    # Получение ответа от конвейера
    res = nlp(QA_input)
    # Загрузка модели и токенайзера
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)
    # Задание входных данных для модели
    question_set = {'question': question, 'context': context}
    # Получение ответа от модели
    res = nlp(question_set)
    # Возврат ответа
    return res.get('answer')

# Пример использования
model_name = "AndrewChar/model-QA-5-epoch-RU"
question = "Что такое API?"
context = "API — описание способов взаимодействия одной компьютерной программы с другими."
answer = test_question_answering(model_name, question, context)
print(answer)
