"""Streamlit app: product-review sentiment classifier.

Serves the models trained in `notebooks/train_review_sentiment.ipynb`:
  * models/lr_pipeline.joblib   TF-IDF + Logistic Regression (fast, explainable)
  * models/transformer/         fine-tuned transformer (optional)
  * models/metadata.json        metrics and settings written by the notebook

Run with:  streamlit run app.py
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))          # so that `src` is importable however the app is launched
from src.text_utils import LABELS, clean_text, explain_linear, light_clean  # noqa: E402

MODELS = BASE / os.environ.get("MODELS_DIR", "models")
ASSETS = BASE / os.environ.get("ASSETS_DIR", "assets")
MAX_BATCH_ROWS = 5000

EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
EXAMPLES = [
    "Absolutely delicious, best coffee I have ever bought. Will order again!",
    "Terrible. Arrived stale and the taste was awful. Not worth the money.",
    "It is okay. Nothing special but not bad either.",
    "Great price, but the packaging was damaged and shipping took two weeks.",
    "Does exactly what it says. Good value.",
]

def _page_title():
    # plain read (no Streamlit call) because set_page_config must be the first Streamlit command
    try:
        return json.loads((MODELS / "metadata.json").read_text()).get("page_title", "Review Sentiment Classifier")
    except Exception:  # noqa: BLE001
        return "Review Sentiment Classifier"


st.set_page_config(page_title=_page_title(), page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_data
def load_metadata():
    path = MODELS / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource(show_spinner="Loading Logistic Regression model...")
def load_lr():
    path = MODELS / "lr_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


def transformer_source():
    """Local folder written by the notebook, or a Hugging Face Hub id in the HF_MODEL_ID env var."""
    env = os.environ.get("HF_MODEL_ID")
    if env:
        return env
    local = MODELS / "transformer"
    has_weights = any((local / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    return str(local) if has_weights else None


@st.cache_resource
def transformer_libs_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource(show_spinner="Loading transformer model (first time only)...")
def load_transformer(source):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    return tok, model


meta = load_metadata()
EXAMPLES = meta.get("examples") or EXAMPLES
ITEM = meta.get("item_name", "review")
lr = load_lr()
tr_source = transformer_source() if transformer_libs_available() else None

if lr is None:
    st.error("`models/lr_pipeline.joblib` not found. Run `notebooks/train_review_sentiment.ipynb` first (section 14 saves the models), "
             "then put the generated `models/` and `assets/` folders in the project root, next to `app.py`.")
    st.stop()

LR_LABEL = "Logistic Regression (TF-IDF): fast, explainable"
TR_LABEL = "Fine-tuned transformer: most accurate"
available = {LR_LABEL: "lr"}
if tr_source:
    available[TR_LABEL] = "transformer"


# ----------------------------------------------------------------------------- prediction
def predict_proba(model_key, texts):
    """Return an (n, 3) array of probabilities with columns in LABELS order."""
    texts = [str(t) for t in texts]
    if model_key == "lr":
        remove_sw = meta.get("lr", {}).get("remove_stopwords", False)
        probs = lr.predict_proba([clean_text(t, remove_stopwords=remove_sw) for t in texts])
        order = [list(lr.classes_).index(label) for label in LABELS]
        return probs[:, order]

    import torch
    tok, model = load_transformer(tr_source)
    max_len = meta.get("transformer", {}).get("max_len", 128)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = [light_clean(t) for t in texts[i:i + 32]]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    probs = np.vstack(out)
    id2label = model.config.id2label
    names = [id2label[i] for i in range(len(id2label))]
    if set(names) == set(LABELS):
        probs = probs[:, [names.index(label) for label in LABELS]]
    return probs


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title(meta.get("app_title", "🛒 Review Sentiment"))
    model_name = st.radio("Model", list(available), help="Both models were trained in the notebook on the same data.")
    model_key = available[model_name]
    if not tr_source:
        st.caption("Transformer not available: no `models/transformer/` folder (or PyTorch/transformers not installed).")

    info = meta.get("lr" if model_key == "lr" else "transformer", {})
    if info.get("test"):
        st.metric("Test macro-F1", f"{info['test']['macro_f1']:.3f}")
        st.metric("Test accuracy", f"{info['test']['accuracy']:.3f}")
        st.caption(f"Prediction speed in training environment: about {info['ms_per_tweet']:.2f} ms per {ITEM}.")

    trained_sk = meta.get("versions", {}).get("scikit-learn")
    if trained_sk and trained_sk != sklearn.__version__:
        st.warning(f"Model trained with scikit-learn {trained_sk}, running {sklearn.__version__}. "
                   "Results may differ; install the same version.")
    st.divider()
    st.caption(meta.get("caption", "Classifies English product reviews as negative, neutral or positive. "
                                   "Trained on star-rated reviews; it can misread sarcasm, and 3-star reviews are often mixed."))


# ----------------------------------------------------------------------------- tabs
tab_single, tab_batch, tab_report = st.tabs([f"Single {ITEM}", "Batch (CSV)", "Model report"])

with tab_single:
    st.subheader(f"Analyse a {ITEM}")
    choice = st.selectbox("Try an example", ["(write your own)"] + EXAMPLES)
    text = st.text_area(f"{ITEM.capitalize()} text", value="" if choice == "(write your own)" else choice,
                        height=140, max_chars=3000, placeholder=f"Type or paste a {ITEM}...")

    if st.button("Analyse", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            probs = predict_proba(model_key, [text])[0]
            pred = LABELS[int(np.argmax(probs))]

            left, right = st.columns([1, 2])
            with left:
                st.metric("Prediction", f"{EMOJI[pred]} {pred.title()}")
                st.metric("Confidence", f"{probs.max():.1%}")
            with right:
                st.caption("Probability per class")
                st.bar_chart(pd.DataFrame({"probability": probs}, index=[l.title() for l in LABELS]))

            if model_key == "lr":
                remove_sw = meta.get("lr", {}).get("remove_stopwords", False)
                _, _, _, contribs = explain_linear(lr, clean_text(text, remove_stopwords=remove_sw), top_k=8)
                st.subheader("Why this prediction?")
                if contribs:
                    st.caption(f"Words that pushed the model towards **{pred}** (positive) or away from it (negative):")
                    st.bar_chart(pd.Series(dict(contribs), name="contribution"))
                else:
                    st.info("None of the words in this text are known to the model, so it relies on the bias term.")
            else:
                st.caption("Word-level explanations are available for the Logistic Regression model. "
                           "Switch the model in the sidebar to see them.")

with tab_batch:
    st.subheader(f"Classify many {ITEM}s from a CSV")
    st.caption(f"Upload a CSV with a column of {ITEM} texts (first {MAX_BATCH_ROWS:,} rows are used).")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the CSV: {exc}")
            data = None
        if data is not None and len(data):
            text_cols = [c for c in data.columns if pd.api.types.is_string_dtype(data[c])] or list(data.columns)
            col = st.selectbox(f"Column with the {ITEM} text", text_cols)
            if st.button("Run batch prediction"):
                subset = data.head(MAX_BATCH_ROWS).copy()
                subset[col] = subset[col].fillna("").astype(str)
                with st.spinner(f"Classifying {len(subset):,} {ITEM}s..."):
                    probs = predict_proba(model_key, subset[col].tolist())
                result = subset.copy()
                result["predicted_sentiment"] = [LABELS[i] for i in probs.argmax(axis=1)]
                result["confidence"] = probs.max(axis=1).round(4)
                for j, label in enumerate(LABELS):
                    result[f"p_{label}"] = probs[:, j].round(4)

                st.success(f"Done: {len(result):,} {ITEM}s classified.")
                st.caption("Sentiment distribution")
                st.bar_chart(result["predicted_sentiment"].value_counts().reindex(LABELS).fillna(0))
                st.dataframe(result.head(200))
                st.download_button("Download results as CSV", result.to_csv(index=False).encode("utf-8"),
                                   file_name="sentiment_predictions.csv", mime="text/csv")

with tab_report:
    st.subheader("Model report")
    if not meta:
        st.info("`models/metadata.json` not found; run the notebook to generate it.")
    else:
        ds = meta.get("dataset", {})
        st.write(f"**Task:** {meta.get('task', '-')}  \n"
                 f"**Data:** {ds.get('name', '-')} ({ds.get('total_rows_after_cleaning', '?'):,} {ITEM}s after cleaning)  \n"
                 f"**Trained:** {meta.get('created', '-')}")

        rows = []
        for key in ("lr", "transformer"):
            m = meta.get(key)
            if m and m.get("test"):
                cross = (meta.get("cross_dataset") or {}).get("metrics", {}).get(m["display_name"], {})
                rows.append({"Model": m["display_name"],
                             "Validation macro-F1": round(m["val_macro_f1"], 4),
                             "Test accuracy": round(m["test"]["accuracy"], 4),
                             "Test macro-F1": round(m["test"]["macro_f1"], 4),
                             "Other-dataset macro-F1": round(cross["macro_f1"], 4) if cross else None,
                             f"ms / {ITEM}": round(m["ms_per_tweet"], 2)})
        if rows:
            st.dataframe(pd.DataFrame(rows))
        if (meta.get("cross_dataset") or {}).get("source"):
            st.caption(f"Other dataset: {meta['cross_dataset']['source']} (models were not retrained on it).")

        st.markdown("**Per-class test results**")
        for key in ("lr", "transformer"):
            m = meta.get(key)
            if m and m.get("test"):
                st.caption(m["display_name"])
                st.dataframe(pd.DataFrame(m["test"]["per_class"]).T.round(3))

    for img, caption in [("model_comparison.png", "Validation macro-F1 by model"),
                         ("confusion_matrices.png", "Confusion matrices on the test set"),
                         ("lr_top_words.png", "Words the Logistic Regression model associates with each class")]:
        if (ASSETS / img).exists():
            st.image(str(ASSETS / img), caption=caption)
