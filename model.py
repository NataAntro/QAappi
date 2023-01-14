from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "AndrewChar/model-QA-5-epoch-RU"

# Получение предсказаний
# pipeline вопрос-ответ
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name) # nlp - переменная для хранения созданного конвейера
QA_input = {
    'question': 'Что такое API?',
    'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
}
res = nlp(QA_input) # res - переменная для хранения результата ответа

# Загрузка модели и токенайзера
model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)

#Текст для анализа его контекста
wikiText = """
API — описание способов взаимодействия одной компьютерной программы с другими.
"""

#Определение ввода вопроса
questionSet = {
                'question': 'Что такое API?',
                'context': wikiText
                
                }

#Вызов модели
res = nlp(questionSet)
res.get('answer')
