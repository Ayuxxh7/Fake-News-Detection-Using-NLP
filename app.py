"""
app.py
------
Streamlit web app for Fake News Detection.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import streamlit as st

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

MIN_WORDS = 5            # minimum meaningful words needed for a prediction
UNCERTAIN_BELOW = 0.65   # if the model score is below this, say "uncertain"

# Sample articles for testing. They are written by hand and are NOT inside the training dataset.
SAMPLE_ARTICLES = {
    "Sample 1 (real-style): Metro update": (
        "The state transport corporation said on Monday that the second phase of the metro line "
        "will open next month after the safety commissioner completes inspections. A spokesperson "
        "told reporters that ticket prices will be published on the official website, and the "
        "project cost of about forty billion rupees was approved in last year's budget."
    ),
    "Sample 2 (real-style): Research study": (
        "Researchers at the national institute published a study on air quality in major cities. "
        "The report, based on data collected over five years, found that pollution levels fell "
        "slightly after new emission rules were introduced. Independent experts said more data is "
        "needed before firm conclusions can be drawn."
    ),
    "Sample 3 (real-style): Budget meeting": (
        "The city council met on Thursday to discuss the annual budget. Members approved funding "
        "for road repairs and drainage work, while opposition members asked for a detailed "
        "breakdown of costs. The final document will be released after the next meeting."
    ),
    "Sample 4 (fake-style): Miracle cure": (
        "SHOCKING!!! Doctors are hiding this miracle cure that kills every disease overnight! "
        "A secret herb from a mountain village beats all medicine, but big companies do not want "
        "you to know. Share this with everyone before they delete it!!!"
    ),
    "Sample 5 (fake-style): Free money": (
        "URGENT!!! Government is giving free money to every citizen tonight. Click this link "
        "right now and enter your details to claim your gift. Forward this message to ten friends "
        "or you will lose the offer forever, hurry!!!"
    ),
    "Sample 6 (fake-style): Conspiracy": (
        "You will not believe what insiders just revealed! The media is lying to you and the "
        "elite are secretly controlling everything. Wake up people, the truth is being hidden, "
        "and only this page shows the real story."
    ),
}


# ---------------------------------------------------------------------------
# Helper functions (kept free of Streamlit code so they are easy to test)
# ---------------------------------------------------------------------------
def load_bundle():
    """Load the saved vectorizer and model from model/trained_model.pkl."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run  'python train_model.py'  first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["vectorizer"], bundle["model"]


def load_metrics():
    """Load the measured results saved during training (or None if missing)."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_news(text, vectorizer, model):
    """
    Predict whether the text is FAKE or REAL.
    Returns a dictionary. Raises ValueError for bad input.
    """
    if text is None or not str(text).strip():
        raise ValueError("Please type or paste some news text first.")

    cleaned = preprocess_text(text)
    if len(cleaned.split()) < MIN_WORDS:
        raise ValueError(
            f"The text is too short. Please enter at least {MIN_WORDS} meaningful words "
            "(a headline or a full article)."
        )

    features = vectorizer.transform([cleaned])
    if features.nnz == 0:
        raise ValueError(
            "None of the words in this text are known to the model, so it cannot predict. "
            "Try a longer English news text."
        )

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    p_real = float(probabilities[classes.index("REAL")])
    p_fake = float(probabilities[classes.index("FAKE")])

    label = "REAL" if p_real >= p_fake else "FAKE"
    score = max(p_real, p_fake)
    return {
        "label": label,
        "score": score,
        "uncertain": score < UNCERTAIN_BELOW,
        "p_real": p_real,
        "p_fake": p_fake,
        "cleaned_text": cleaned,
    }


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------
def set_sample(sample_name):
    """Callback: put the chosen sample article into the text box."""
    if sample_name in SAMPLE_ARTICLES:
        st.session_state["news_text"] = SAMPLE_ARTICLES[sample_name]


def clear_text():
    """Callback: clear the text box."""
    st.session_state["news_text"] = ""


def show_result(result):
    """Display the prediction in a coloured box."""
    is_fake = result["label"] == "FAKE"
    color = "#c62828" if is_fake else "#2e7d32"
    background = "#fdecea" if is_fake else "#e8f5e9"
    title = "FAKE NEWS" if is_fake else "REAL NEWS"
    icon = "🚨" if is_fake else "✅"

    st.markdown(
        f"""
        <div style="background:{background}; border-left:8px solid {color};
                    padding:18px 22px; border-radius:8px; margin-top:10px;">
            <span style="font-size:30px; font-weight:700; color:{color};">{icon} {title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(f"**Model score:** {result['score'] * 100:.1f}%")
    st.progress(min(max(result["score"], 0.0), 1.0))
    st.caption(
        "This score is the probability given by Logistic Regression. It is NOT a guarantee "
        "that the news is true or false, and it has not been verified on real-world news."
    )

    if result["uncertain"]:
        st.warning("The model is not very sure about this text. Please verify it from trusted sources.")

    with st.expander("See the text after preprocessing"):
        st.write(result["cleaned_text"])


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

    st.title("📰 Fake News Detection using NLP")
    st.write(
        "Paste a news headline or article below and click **Predict**. "
        "The app uses TF-IDF and Logistic Regression to classify it as **REAL** or **FAKE**."
    )

    # Load the model (stop the app with a clear message if it is missing/broken)
    @st.cache_resource
    def get_model():
        return load_bundle()

    try:
        vectorizer, model = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:  # e.g. scikit-learn version mismatch
        st.error(
            "The saved model could not be loaded. This usually happens when scikit-learn on this "
            "computer is a different version from the one used for training. "
            "Please run  'python train_model.py'  again on this computer.\n\n"
            f"Details: {error}"
        )
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("About this project")
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Features:** TF-IDF (unigrams + bigrams)")
        metrics = load_metrics()
        if metrics:
            st.subheader("Measured test results")
            st.write(f"Accuracy: **{metrics['accuracy']:.2%}**")
            st.write(f"Precision: **{metrics['precision']:.2%}**")
            st.write(f"Recall: **{metrics['recall']:.2%}**")
            st.write(f"F1 score: **{metrics['f1_score']:.2%}**")
            st.caption(
                f"Measured on a held-out test set of {metrics['test_rows']} rows from "
                f"'{metrics['dataset_file']}'. If this is the small sample dataset, the numbers "
                "are optimistic and do not show real-world performance."
            )
        st.subheader("Limitations")
        st.write(
            "- The model learns writing style and words, not facts.\n"
            "- It cannot check if a claim is actually true.\n"
            "- Only English text is supported."
        )

    # Sample articles
    st.subheader("Try a sample")
    sample_name = st.selectbox("Choose a sample article", list(SAMPLE_ARTICLES.keys()))
    col1, col2 = st.columns(2)
    col1.button("Load sample", on_click=set_sample, args=(sample_name,), use_container_width=True)
    col2.button("Clear text", on_click=clear_text, use_container_width=True)

    # Input box
    st.subheader("Enter news text")
    st.text_area(
        "News headline or article",
        key="news_text",
        height=220,
        placeholder="Paste your news article or headline here...",
    )

    # Predict
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        try:
            result = predict_news(st.session_state.get("news_text", ""), vectorizer, model)
            show_result(result)
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(f"Something went wrong while predicting: {error}")

    st.divider()
    st.caption("College mini project - Fake News Detection using NLP and Machine Learning")


if __name__ == "__main__":
    main()
