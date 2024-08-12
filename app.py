import streamlit as st
from executors.inferencer import Inferencer
from configs.experiment_config import experiment_cfg


def run_inference(prompt):
    inferencer = Inferencer(experiment_cfg)
    result = inferencer.predict(prompt)
    return result


# Интерфейс Streamlit
st.title('Tiny LLM')

# Поля ввода
random_sentence = st.text_input('Random Sentence (optional):', '')
short_summary = st.text_input('Short Summary (optional):', '')
words = st.text_input('Words (optional):', '')

# Чекбоксы для выбора фич
option1 = st.checkbox('Dialogue')
option2 = st.checkbox('BadEnding')
option3 = st.checkbox('MoralValue')
option4 = st.checkbox('Twist')
option5 = st.checkbox('Foreshadowing')
option6 = st.checkbox('Conflict')

# Формируем список выбранных опций
features = []
if option1:
    features.append('Dialogue')
if option2:
    features.append('BadEnding')
if option3:
    features.append('MoralValue')
if option4:
    features.append('Twist')
if option5:
    features.append('Foreshadowing')
if option6:
    features.append('Conflict')

# Объединяем текст из всех не пустых полей
prompt_parts = []
if random_sentence:
    prompt_parts.append("Random sentence: " + random_sentence)
if short_summary:
    prompt_parts.append("Summary: " + short_summary)
if words:
    prompt_parts.append("Words: " + words)
if features:
    prompt_parts.append("Features: " + ', '.join(features))

# Создаем окончательный промпт
final_prompt = ' '.join(prompt_parts) + " Story:"

if final_prompt:
    # st.write(f"Промпт для модели: {final_prompt}")

    if st.button('Предсказать'):
        result = run_inference(final_prompt)
        st.write('Результат:', result)
else:
    st.write("Заполните хотя бы одно поле для создания промпта.")