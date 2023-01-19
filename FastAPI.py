from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "AndrewChar/model-QA-5-epoch-RU"

# Получение предсказаний
# pipeline вопрос-ответ
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name) # nlp - переменная для хранения созданного конвейера

# Загрузка модели и токенайзера
model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.post("/qa")
async def question_answering(request: Request):
    data = await request.json()

    question = data["question"]
    context = data["context"]

    res = nlp({'question': question, 'context': context})

    return JSONResponse(content={"answer": res.get("answer")}, status_code=status.HTTP_200_OK)


