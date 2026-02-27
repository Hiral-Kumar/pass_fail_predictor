import streamlit as st
import pandas as pd
import joblib

PASSING_PERCENTAGE = 45

import os
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "pass_fail_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Train model on startup (deployment-safe)
    X = df[feature_columns]
    y = df["final_result"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

subjects = [
    "maths", "science", "english",
    "hindi", "social_science", "computer_science"
]

st.set_page_config(page_title="Pass/Fail Predictor", layout="centered")
st.markdown("# 🎓 Student Pass/Fail Predictor")
st.markdown("Enter your marks and attendance to get a prediction, along with some key advice")

# three‑column layout for subjects
input_data = {}
cols = st.columns(3)
for idx, subject in enumerate(subjects):
    with cols[idx % 3]:
        input_data[f"{subject}_total"] = st.number_input(
            f"{subject.capitalize()} Total Marks", 0, 100, 50
        )

attendance = st.slider("Attendance Score (0–10)", 0.0, 10.0, 7.0)
input_data["attendance_score"] = attendance
input_data["final_weighted_score"] = (
    0.85 * sum(input_data[f"{s}_total"] for s in subjects) / len(subjects)
    + 0.15 * (attendance * 10)
)

def analyze_input(data):
    failed = []
    near = []
    for subject in subjects:
        score = data[f"{subject}_total"]
        if score < PASSING_PERCENTAGE:
            failed.append(subject)
        elif PASSING_PERCENTAGE <= score < PASSING_PERCENTAGE + 10:
            near.append(subject)
    reason_parts = []
    tips = []
    if failed:
        reason_parts.append("Failed: " + ", ".join(failed))
        tips.append("Work with a mentor on " + ", ".join(failed))
    if near:
        reason_parts.append("Near fail: " + ", ".join(near))
        tips.append("Boost understanding in " + ", ".join(near))
    if data["attendance_score"] < 6:
        reason_parts.append("Low attendance")
        tips.append("Improve attendance; it influences your score")
    return " | ".join(reason_parts), " | ".join(tips)

if st.button("Predict"):
    df_input = pd.DataFrame([input_data])

    # enforce per-subject pass criteria regardless of model
    subject_failures = [s for s in subjects
                        if input_data[f"{s}_total"] < PASSING_PERCENTAGE]
    if subject_failures:
        pred = 0
        # set risk very high when at least one subject failed
        risk_percent = 100.0
        reason = "Failed subjects: " + ", ".join(subject_failures)
        tips = "Work with a mentor on " + ", ".join(subject_failures)
    else:
        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        risk_percent = round((1 - prob) * 100, 2)
        reason, tips = analyze_input(input_data)

    st.subheader("Result")
    if pred == 1:
        st.success("PASS ✅")
    else:
        st.error("FAIL ❌")

    st.markdown(f"**Risk of failing again:** {risk_percent}%")
    if reason:
        st.markdown(f"**Reason identified:** {reason}")
    if tips:
        st.markdown(f"**Suggestions:** {tips}")

    st.markdown("---")
    st.markdown("### 📘 General Tips")
    st.write("- Keep attendance high; it is part of the final score.")
    st.write("- Practice regularly and solve past question papers.")
    st.write("- Consult your teacher or mentor for difficult subjects.")

# dataset preview (for development)
#try:
# df = pd.read_csv("student_pass_fail_dataset_10000.csv")
#st.subheader("📂 Dataset Preview")
   # st.dataframe(df.head())
#except FileNotFoundError:
 #   st.warning("Dataset not found. Please upload the dataset.")

#st.subheader("✅ Deployment Status")
#st.success("Streamlit app is running successfully!")

