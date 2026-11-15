import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. Chargement du modèle
# ============================================================

MODEL_PATH = Path(__file__).parent / "logistic_regression_model.joblib"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Mental Health Risk Prediction")

st.write(
    "Enter the student's information to predict the mental health risk level."
)


# ============================================================
# 3. Formulaire
# ============================================================

with st.form("prediction_form"):

    st.subheader("Student Information")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Colonne 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=30,
            value=20,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Non-binary"]
        )

        year_of_study = st.number_input(
            "Year of Study",
            min_value=1,
            max_value=5,
            value=2,
            step=1
        )

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=3.5,
            max_value=9.5,
            value=7.0,
            step=0.1
        )

        study_hours = st.number_input(
            "Study Hours Per Week",
            min_value=5.0,
            max_value=55.0,
            value=20.0,
            step=0.5
        )

    # --------------------------------------------------------
    # Colonne 2
    # --------------------------------------------------------

    with col2:

        gpa = st.number_input(
            "GPA",
            min_value=1.5,
            max_value=4.0,
            value=3.0,
            step=0.1
        )

        social_media_hours = st.number_input(
            "Social Media Hours",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.1
        )

        screen_time_hours = st.number_input(
            "Screen Time Hours",
            min_value=1.0,
            max_value=14.0,
            value=6.0,
            step=0.1
        )

        physical_activity = st.selectbox(
            "Physical Activity",
            ["Low", "Moderate", "High"]
        )

        financial_stress = st.number_input(
            "Financial Stress",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

    # --------------------------------------------------------
    # Colonne 3
    # --------------------------------------------------------

    with col3:

        family_support = st.number_input(
            "Family Support",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        attendance_rate = st.number_input(
            "Attendance Rate (%)",
            min_value=30.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

        part_time_job = st.selectbox(
            "Part Time Job",
            ["Yes", "No"]
        )

        diet_quality = st.selectbox(
            "Diet Quality",
            ["Poor", "Average", "Good"]
        )

        previous_mental_health_history = st.selectbox(
            "Previous Mental Health History",
            ["Yes", "No"]
        )

    # --------------------------------------------------------
    # Bouton
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "Predict Mental Health Risk"
    )


# ============================================================
# 4. Prédiction
# ============================================================

if submitted:

    # Création d'une nouvelle observation
    new_observation = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Year_of_Study": [year_of_study],
        "Sleep_Hours": [sleep_hours],
        "Study_Hours_Per_Week": [study_hours],
        "GPA": [gpa],
        "Social_Media_Hours": [social_media_hours],
        "Screen_Time_Hours": [screen_time_hours],
        "Physical_Activity": [physical_activity],
        "Financial_Stress": [financial_stress],
        "Family_Support": [family_support],
        "Attendance_Rate": [attendance_rate],
        "Part_Time_Job": [part_time_job],
        "Diet_Quality": [diet_quality],
        "Previous_Mental_Health_History": [previous_mental_health_history]
    })

    # Prédiction
    prediction = model.predict(new_observation)[0]

    # ========================================================
    # Affichage de la prédiction
    # ========================================================

    st.subheader("Prediction")

    st.success(
        f"Predicted Mental Health Risk: **{prediction}**"
    )

    # ========================================================
    # Probabilités
    # ========================================================

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            new_observation
        )[0]

        classes = model.classes_

        probability_df = pd.DataFrame({
            "Risk Level": classes,
            "Probability": probabilities
        })

        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        ).round(2)

        probability_df = probability_df[
            ["Risk Level", "Probability (%)"]
        ]

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        st.bar_chart(
            probability_df.set_index("Risk Level")
        )