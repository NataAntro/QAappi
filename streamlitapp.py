# Импорт модулей
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Настройки страницы
st.set_page_config(page_title="My QA App", page_icon=":guardsman:", layout="wide", initial_sidebar_state="auto")

# Имя и инициализация модели 
model_name = "AndrewChar/model-QA-5-epoch-RU"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Что такое API?',
    'context': 'API — описание способов взаимодействия одной компьютерной программы с другими.'
}
res = nlp(QA_input) # res - переменная для хранения результата ответа

# Загрузка модели и токенайзера
model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)

# Текст и вопрос для анализа
context = st.text_area("Вставить анализируемый текст:")
question = st.text_input("Задать вопрос из контекста:")

# Предупреждение, если вопрос не задан
if question.strip() == "":
    st.warning("Пожалуйста, введите корректный вопрос.")
# Подготовка данных, анализ, стили для отображения ответа, ответ
else:
    QA_input = {'question': question, 'context': context}
    res = nlp(QA_input)
    answer = res.get('answer')
    st.markdown("""
    <style>
    .answer-box {
        border: 1px solid #ccc;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='answer-box'>Ответ нейросети: {}</div>".format(answer), unsafe_allow_html=True)
