import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
from datetime import datetime
import json
import os

# Set page config as first Streamlit command
st.set_page_config(page_title="NER Tagger", layout="wide")

# Log file path
LOG_FILE = "logs.jsonl"

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("amruthsaravanan/NLP_Coursework_2")
    model = AutoModelForTokenClassification.from_pretrained("amruthsaravanan/NLP_Coursework_2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ✅ Updated token mapping to 5 tokens
id_to_label = {0: "O", 1: "B-AC", 2: "I-AC", 3: "B-LF", 4: "I-LF"}
color_map = {
    "B-AC": "#FFD700",
    "I-AC": "#FFA07A",
    "B-LF": "#ADFF2F",
    "I-LF": "#87CEFA",
    "O": "#E0E0E0"
}

st.title("🧠 BERT Token Classification for NER")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Tabs Layout ----
tab1, tab2 = st.tabs(["🔠 Predict", "📜 History"])

with tab1:
    st.subheader("Enter your sentence")
    default_text = "The WHO announced a new policy on air quality."
    user_input = st.text_area("Text:", value=default_text, height=100)

    st.caption(f"📝 Characters: {len(user_input)} | Words: {len(user_input.split())}")

    if st.button("🚀 Predict"):
        if user_input.strip():
            words = user_input.split()
            inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True,
                               truncation=True, padding=True)
            with st.spinner("Classifying..."):
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=2)

            # Align predictions
            word_ids = inputs.word_ids(batch_index=0)
            word_to_label = {}
            for idx, wid in enumerate(word_ids):
                if wid is not None and wid not in word_to_label:
                    word_to_label[wid] = id_to_label[predictions[0][idx].item()]
            results = [(w, word_to_label.get(i, "O")) for i, w in enumerate(words)]

            # Show colored output
            def render_colored(tokens):
                html = ""
                for word, label in tokens:
                    html += f"<span style='background-color:{color_map[label]}; padding:5px 8px; margin:4px; border-radius:5px; display:inline-block'>{word} ({label})</span> "
                return html

            st.markdown("### Prediction")
            st.markdown(render_colored(results), unsafe_allow_html=True)

            # Save to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_str = ", ".join(f"{w}:{lbl}" for w, lbl in results)
            st.session_state.history.append((user_input, result_str, timestamp))

            # Save to JSONL log file
            log_entry = {
                "timestamp": timestamp,
                "input": user_input,
                "predictions": results
            }
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                    log_f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                st.error(f"Logging failed: {e}")

            # Download CSV
            df = pd.DataFrame(results, columns=["Word", "Label"])
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download as CSV", csv, file_name="ner_results.csv", mime="text/csv")

with tab2:
    st.subheader("Prediction History")
    if st.session_state.history:
        for i, (inp, out, ts) in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"#{len(st.session_state.history) - i + 1} • {ts}"):
                st.markdown(f"**Input:** {inp}")
                st.markdown(f"**Prediction:** {out}")
    else:
        st.info("No prediction history yet.")

    if st.button("🧹 Clear History"):
        st.session_state.history = []

    # Download log file
    st.markdown("---")
    st.markdown("#### Download Log")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📄 Download Logs (JSONL)", f, file_name="ner_logs.jsonl", mime="application/json")
    else:
        st.caption("No logs available yet.")
