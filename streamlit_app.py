import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.set_page_config(page_title="Emotion AI", page_icon="🧠")

st.title("🧠 Emotion Analyzer")
st.write("Type something and I will detect your emotion 💖")

# Input
text = st.text_area("Enter your text here:")

# Button
if st.button("Analyze Emotion"):
    if text.strip() != "":
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        # Display result
        if prediction == "joy":
            st.success(f"😊 Emotion: {prediction}")
        elif prediction == "sadness":
            st.info(f"😔 Emotion: {prediction}")
        elif prediction == "anger":
            st.error(f"😡 Emotion: {prediction}")
        else:
            st.write(f"Emotion: {prediction}")

    else:
        st.warning("⚠ Please enter some text")
st.markdown("---")
st.markdown("✨ AI powered emotion detection system")