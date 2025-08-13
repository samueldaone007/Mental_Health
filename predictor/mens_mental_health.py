import streamlit as st
import pandas as pd
import joblib
import openai

# ───────────────────────
# 🔐 Set up Groq API
# ───────────────────────
openai.api_key = "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"  # ← Replace this with your real key
openai.api_base = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint

# ───────────────────────
# 🧠 Load model & encoder
# ───────────────────────
encoder = joblib.load('encoder_pipeline.pkl')
model = joblib.load('model.pkl')  # Your trained mental health prediction model

# ───────────────────────
# 🎛 Input Fields
# ───────────────────────
fields = [
    "Sadness", "Euphoric", "Exhausted", "Sleep dissorder", "Mood Swing",
    "Suicidal thoughts", "Anorxia", "Authority Respect", "Try-Explanation",
    "Aggressive Response", "Ignore & Move-On", "Nervous Break-down",
    "Admit Mistakes", "Overthinking", "Sexual Activity", "Concentration", "Optimisim"
]
rating_fields = ["Sexual Activity", "Concentration", "Optimisim"]

# ───────────────────────
# 🧾 Form UI
# ───────────────────────
st.title("🧠 Mental Health Diagnosis Predictor")
st.write("Fill in the symptoms and behaviors below to get a predicted mental health diagnosis.")

with st.form("diagnosis_form"):
    input_data = {}
    for field in fields:
        if field in rating_fields:
            input_data[field] = st.selectbox(field, [
                "1 From 10", "2 From 10", "3 From 10", "4 From 10", "5 From 10",
                "6 From 10", "7 From 10", "8 From 10", "9 From 10", "10 From 10"])
        else:
            input_data[field] = st.selectbox(field, ["YES", "NO", "Seldom", "Usually", "Sometimes", "Most-Often"])

    submit = st.form_submit_button("Predict Diagnosis")

# ───────────────────────
# 🔮 Prediction Logic
# ───────────────────────
if submit:
    input_df = pd.DataFrame([input_data])
    X_encoded = encoder.transform(input_df)
    prediction = model.predict(X_encoded)

    if prediction == 0:
        st.success("🩺 Predicted Diagnosis: Bipolar Type 1")
    elif prediction == 1:
        st.success("🩺 Predicted Diagnosis: Bipolar Type 2")
    else:
        st.success("🩺 Predicted Diagnosis: Depression")

# ───────────────────────
# 💬 Groq Chatbot
# ───────────────────────
st.sidebar.title("💬 Mental Health Chatbot")
st.sidebar.write("Ask about symptoms, disorders, or mental health support.")

user_question = st.sidebar.text_input("Ask a question:")

if user_question:
    prompt = f"""
    You are a helpful assistant in a men's mental health support app.
    You only answer questions related to mental illness, symptoms in the dataset, diagnoses (like Bipolar or Depression),
    and helpful mental wellness advice. Stay focused on mental health topics. Keep your replies not too short but not too long.

    Question: {user_question}
    """

    try:
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",  # Groq model
            messages=[
                {"role": "system", "content": "You are a helpful mental health assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message["content"].strip()
        st.sidebar.markdown(f"**Bot:** {reply}")
    except Exception as e:
        st.sidebar.error(f"❌ Error from Groq API: {e}")

