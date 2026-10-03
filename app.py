import csv
import html
import io
import re

import joblib
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer


st.set_page_config(page_title="Comment Safety Lab", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #172b2a;
        --muted: #61716d;
        --paper: #f4f5ef;
        --panel: #ffffff;
        --line: #dce3dc;
        --green: #176b55;
        --green-soft: #e5f3ec;
        --red: #b3413e;
        --red-soft: #fff0ec;
        --lime: #d9f36a;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
    .hero {
        padding: 1.5rem 1.8rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(110deg, #e7f1e9 0%, #fff 62%, #f4f1df 100%);
        margin-bottom: 1.3rem;
    }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(1.8rem, 4vw, 2.6rem); margin: .35rem 0 .3rem; }
    .hero p { color: var(--muted); margin: 0; }
    [data-testid="stTextArea"] textarea { background: #fff; border-color: #b8c8bf; border-radius: 6px; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius: 5px; border: 1px solid var(--green); font-weight: 700;
        min-height: 2.7rem;
    }
    [data-testid="stButton"] button[kind="primary"] { background: var(--green); color: white; }
    .result {
        border: 1px solid var(--line); border-left: 5px solid var(--green);
        border-radius: 6px; background: var(--panel); padding: 1rem 1.1rem;
        margin: .65rem 0;
    }
    .result.toxic { border-left-color: var(--red); }
    .result-label { font-weight: 700; color: var(--green); }
    .result-label.toxic { color: var(--red); }
    .result-comment { color: var(--ink); margin: .45rem 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
    .stProgress > div > div > div > div { background-color: var(--green); }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 3rem; }
        .hero { padding: 1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load("toxic_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer, WordNetLemmatizer()


try:
    model, vectorizer, lemmatizer = load_assets()
except FileNotFoundError:
    st.error("Model files are missing. Run `train_model.py` to create them.")
    st.stop()


def clean_text(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def use_example(text):
    st.session_state["comments_input"] = text


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Comment Safety Lab · Binary classifier</div>
      <h1>Read the tone. Keep the conversation.</h1>
      <p>Screen one comment or a whole batch, then tune how cautious the flagging should be.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Detection settings")
    threshold = st.slider(
        "Toxic flag threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower values catch more potentially toxic comments but may flag more harmless ones.",
    )
    st.caption("This changes the decision tradeoff, not the model's underlying probabilities.")

st.subheader("Analyze comments")
st.caption("Paste comments below, one per line. Blank lines are ignored.")
example_cols = st.columns(3)
example_texts = [
    "Thanks for taking the time to explain your point.",
    "You are an idiot and nobody wants you here.",
    "I disagree with this decision. Please explain the reasoning.",
]
example_labels = ["Try a friendly example", "Try a toxic example", "Try a disagreement"]
for column, label, sample in zip(example_cols, example_labels, example_texts):
    column.button(label, on_click=use_example, args=(sample,), use_container_width=True)

with st.form("analysis_form"):
    comments_input = st.text_area(
        "Comments",
        key="comments_input",
        height=190,
        placeholder="Paste one or more comments here...",
    )
    submitted = st.form_submit_button("Analyze comments", type="primary", use_container_width=True)

if submitted:
    comments = [line.strip() for line in comments_input.splitlines() if line.strip()]
    if not comments:
        st.warning("Add at least one non-empty comment to analyze.")
    elif len(comments) > 100:
        st.warning("For a fast, readable analysis, submit up to 100 comments at a time.")
    else:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        with st.spinner(f"Analyzing {len(comments)} comment{'s' if len(comments) != 1 else ''}..."):
            cleaned_comments = [clean_text(comment) for comment in comments]
            features = vectorizer.transform(cleaned_comments)
            class_values = list(model.classes_)
            toxic_class_index = class_values.index(1)
            toxic_probabilities = model.predict_proba(features)[:, toxic_class_index]

        flagged = 0
        result_rows = []
        for comment, toxic_probability in zip(comments, toxic_probabilities):
            is_toxic = toxic_probability >= threshold
            flagged += int(is_toxic)
            label = "Toxic" if is_toxic else "Non-toxic"
            confidence = toxic_probability if is_toxic else 1 - toxic_probability
            result_rows.append((comment, label, confidence, toxic_probability))

        metric_cols = st.columns(3)
        metric_cols[0].metric("Comments checked", len(comments))
        metric_cols[1].metric("Flagged toxic", flagged)
        metric_cols[2].metric("Below threshold", len(comments) - flagged)
        st.progress(flagged / len(comments), text=f"{flagged} of {len(comments)} flagged at {threshold:.0%}")

        st.subheader("Results")
        for index, (comment, label, confidence, toxic_probability) in enumerate(result_rows, 1):
            toxic_class = label == "Toxic"
            label_class = "toxic" if toxic_class else ""
            box_class = "toxic" if toxic_class else ""
            st.markdown(
                f"<article class='result {box_class}'>"
                f"<div class='result-label {label_class}'>"
                f"{index:02d} &nbsp; {label} · {confidence:.0%} confidence</div>"
                f"<p class='result-comment'>{html.escape(comment)}</p></article>",
                unsafe_allow_html=True,
            )
            st.progress(float(toxic_probability), text=f"Toxic probability: {toxic_probability:.0%}")

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["comment", "prediction", "confidence", "toxic_probability", "threshold"])
        writer.writerows(
            (comment, label, f"{confidence:.6f}", f"{probability:.6f}", f"{threshold:.2f}")
            for comment, label, confidence, probability in result_rows
        )
        st.download_button(
            "Download results as CSV",
            data=csv_buffer.getvalue(),
            file_name="toxic_comment_results.csv",
            mime="text/csv",
        )

st.caption("Predictions are automated estimates. Review context before taking action on a comment.")