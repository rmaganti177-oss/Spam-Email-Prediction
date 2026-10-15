import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Ensure page config is first Streamlit command
st.set_page_config(
    page_title="AI Spam Detector",
    layout="wide",
    page_icon="📧",
)

# ✅ Load trained model and vectorizer from 'models' folder
@st.cache_resource
def load_model():
    with open("models/spam_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

model = load_model()
vectorizer = load_vectorizer()

# ✅ Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/SpamEmail.svg/800px-SpamEmail.svg.png", width=150)
    st.title("📧 AI Spam Detector")
    st.info("Enter text or upload a file to detect spam emails.")

# ✅ Main UI
st.markdown("<h2 style='text-align: center;'>🚀 AI-Powered Spam Mail Detector</h2>", unsafe_allow_html=True)
st.write("### 🔍 Detect whether an email is spam or not using AI!")

# ✅ User Input
email_text = st.text_area("📩 Enter email text:", height=150)

uploaded_file = st.file_uploader("📂 Or upload a `.txt` file", type=["txt"])
if uploaded_file is not None:
    email_text = uploaded_file.getvalue().decode("utf-8")

# ✅ Process Input
if st.button("🔍 Detect Spam", use_container_width=True):
    if email_text.strip():
        # Convert text to TF-IDF features
        email_vector = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector)[0][1]  # Spam probability
        
        # Display result
        st.subheader("🔹 Result:")
        if prediction == 1:
            st.error(f"🚨 **Spam Detected!** (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ **Not Spam!** (Confidence: {(1 - confidence):.2%})")

    else:
        st.warning("⚠️ Please enter text or upload a file.")

# ✅ Batch Processing for Multiple Emails
st.markdown("---")
st.subheader("📋 Bulk Email Spam Detection")
uploaded_csv = st.file_uploader("📂 Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    
    if "text" not in df.columns:
        st.error("❌ CSV file must have a 'text' column.")
    else:
        df["Prediction"] = model.predict(vectorizer.transform(df["text"]))
        df["Confidence"] = model.predict_proba(vectorizer.transform(df["text"]))[:, 1]
        df["Result"] = df["Prediction"].apply(lambda x: "Spam" if x == 1 else "Not Spam")

        st.dataframe(df[["text", "Result", "Confidence"]])

        # ✅ Downloadable CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📂️ Download Results", csv_data, "spam_detection_results.csv", "text/csv")

# ✅ Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Developed with ❤️ by Ujwal Reddy</div>",
    unsafe_allow_html=True,
)