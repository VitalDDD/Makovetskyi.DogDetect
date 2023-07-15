
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Завантажуємо модель
model = tf.saved_model.load('dog_breed_model')

# Завантажуємо список назв порід собак
breed_names = ['Бігль', 'Бульдог', 'Кокер-спаніель', 'Далматинець', 'Німецька вівчарка', 'Хаскі', 'Лабрадор', 'Пудель', 'Ротвеллер']

# Функція для передбачення за допомогою моделі
def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    prediction = model(image)
    predicted_class = np.argmax(prediction)
    predicted_breed = breed_names[predicted_class]
    return predicted_breed

# Заголовок сторінки
st.title('Розпізнавання порід собак')

# Завантаження фото
uploaded_file = st.file_uploader('Завантажте фото собаки', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене фото', use_column_width=True)

    # Виконуємо передбачення при натисканні кнопки
    if st.button('Почати розпізнавання'):
        prediction = predict(image)
        st.write('Результати передбачення:')
        st.write(prediction)
