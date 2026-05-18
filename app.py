import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App", page_icon="🩺", layout="centered"
)

# 1. Load the trained model safely


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Ошибка: Файл 'model.pkl' не найден. Убедитесь, что сначала запустили скрипт обучения модели."
    )
    st.stop()

# 2. UI Layout
st.title("🩺 Diabetes Prediction Web App")
st.write(
    "Заполните клинические показатели ниже, чтобы получить прогноз модели."
)
st.write("---")

# Split input fields into two columns for a cleaner interface
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies (Беременности)", min_value=0, max_value=20, value=1, step=1
    )
    glucose = st.number_input(
        "Glucose (Глюкоза)", min_value=0, max_value=300, value=120, step=1
    )
    blood_pressure = st.number_input(
        "Blood Pressure (Давление)", min_value=0, max_value=200, value=70, step=1
    )
    skin_thickness = st.number_input(
        "Skin Thickness (Толщина складки кожи)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
    )

with col2:
    insulin = st.number_input(
        "Insulin (Инсулин)", min_value=0, max_value=900, value=80, step=1
    )
    bmi = st.number_input(
        "BMI (Индекс массы тела)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        format="%.1f",
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        format="%.3f",
    )
    age = st.number_input(
        "Age (Возраст)", min_value=1, max_value=120, value=33, step=1
    )

st.write("---")

# 3. Model Prediction Logic
# WARNING: Features MUST be in the exact same order as your x_train DataFrame columns!
input_data = pd.DataFrame(
    [
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
    ]
)

# Button to trigger prediction
if st.button("Рассчитать прогноз", type="primary"):
    # Make predictions
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Результат анализа:")

    if prediction[0] == 1:
        st.error(
            f"⚠️ **Высокий риск:** Модель прогнозирует наличие диабета с вероятностью **{prediction_proba[0][1]:.2%}**."
        )
    else:
        st.success(
            f"✅ **Низкий риск:** Модель прогнозирует отсутствие диабета с вероятностью **{prediction_proba[0][0]:.2%}**."
        )
