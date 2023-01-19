from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import uvicorn

app = FastAPI()

@app.post("/qa")
async def question_answering(request: Request):
    
    data = await request.json()
    
    question = data["question"]
    context = data["context"]
   
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    res = nlp({'question': question, 'context': context})
    
    return JSONResponse(content={"answer": res.get("answer")}, status_code=status.HTTP_200_OK)
