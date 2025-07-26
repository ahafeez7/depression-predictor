import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Depression/Anxiety Risk Predictor")
st.markdown("Enter your details to get a risk estimate:")

# User inputs
age = st.slider("Age", 18, 90, 35)
is_female = st.selectbox("Gender", ["Male", "Female"]) == "Female"
ssri_count = st.slider("Number of Antidepressants Prescribed", 0, 20, 1)
phq9_score = st.slider("Max PHQ-9 Score", 0, 27, 7)
sleep = st.slider("Average Sleep Hours", 0.0, 12.0, 7.0)
is_smoker = st.selectbox("Are you a current/former smoker?", ["No", "Yes"]) == "Yes"
exercise = st.slider("Exercise Days per Week", 0, 7, 3)

# Race (one-hot simplified)
race = st.selectbox("Race", ["White", "Black", "Asian"])
race_black = 1 if race == "Black" else 0
race_asian = 1 if race == "Asian" else 0

# Prepare input row
X_input = pd.DataFrame([[
    2025 - age, int(is_female), ssri_count, phq9_score, sleep, int(is_smoker), exercise,
    race_black, race_asian
]], columns=['year_of_birth', 'is_female', 'ssri_count', 'max_phq9_score',
             'avg_sleep_hours', 'is_smoker', 'exercise_days_per_week',
             'race_concept_id_8527', 'race_concept_id_8657'])

# Scale numeric features
X_input[['avg_sleep_hours', 'exercise_days_per_week']] = scaler.transform(
    X_input[['avg_sleep_hours', 'exercise_days_per_week']]
)

# Prediction
pred_prob = model.predict_proba(X_input)[:, 1][0]
st.subheader("Predicted Risk of Depression/Anxiety:")
st.metric("Risk Probability", f"{pred_prob:.2%}")

if pred_prob > 0.5:
    st.warning("⚠️ High Risk: Consider consulting a healthcare provider.")
else:
    st.success("✅ Low Risk")
