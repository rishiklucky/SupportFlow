import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import numpy as np
import subprocess
from pathlib import Path
from typing import Tuple, Optional

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CATEGORY_MODEL_PATH = MODEL_DIR / "category_model.pkl"
URGENCY_MODEL_PATH = MODEL_DIR / "urgency_model.pkl"
DATA_FILE = BASE_DIR / "support_tickets_data.csv"

# Page Config
st.set_page_config(
    page_title="SupportFlow AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# UI theme (minimal)
def apply_dark_theme():
    st.markdown(
        """
    <style>
    :root {
        --bg-primary: #111111;
        --bg-secondary: #1e1e1e;
        --text-primary: #ffffff;
        --text-secondary: #e0e0e0;
        --border: #333333;
        --accent: #00d4ff;
    }
    body { background-color: #111111; color: #ffffff; }
    .stButton > button { background-color: #00d4ff; color: #111111; border: none; }
    </style>
    """,
        unsafe_allow_html=True,
    )

# Load models with caching
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[object]]:
    """
    Attempts to load serialized models from the models/ directory.
    Returns (category_model, urgency_model) or (None, None) on failure.
    """
    if not CATEGORY_MODEL_PATH.exists() or not URGENCY_MODEL_PATH.exists():
        return None, None
    try:
        cat_model = joblib.load(CATEGORY_MODEL_PATH)
        urg_model = joblib.load(URGENCY_MODEL_PATH)
        return cat_model, urg_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

def train_models_via_script():
    """
    Run train_models.py to (re)create models. Uses subprocess to ensure a fresh process.
    """
    train_py = BASE_DIR / "train_models.py"
    if not train_py.exists():
        st.error("train_models.py not found in project root.")
        return False

    with st.spinner("Training models (this can take a few minutes)..."):
        try:
            # Use same python interpreter
            completed = subprocess.run([os.sys.executable, str(train_py)], check=True, capture_output=True, text=True)
            st.success("Training finished. Reloading models...")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Model training failed:\n{e.stdout}\n{e.stderr}")
            return False

def ensure_models_loaded():
    cat_model, urg_model = load_models()
    if cat_model is None or urg_model is None:
        st.warning("Models not found. You can train them now (or upload pre-trained .pkl files to the `models/` folder).")
        if st.button("Train models now"):
            ok = train_models_via_script()
            if ok:
                # clear cache and reload
                load_models.clear()
                st.experimental_rerun()
        return None, None
    return cat_model, urg_model

# Prediction helpers
def safe_predict_proba(model, texts):
    """
    Returns (classes, probs) where probs is an array of probability distributions for each input.
    If model doesn't support predict_proba, attempt to derive probabilities from decision_function using softmax.
    """
    import numpy as _np
    from scipy.special import softmax

    try:
        probs = model.predict_proba(texts)
        classes = model.classes_
        return classes, probs
    except Exception:
        try:
            decisions = model.decision_function(texts)
            # decision_function shape handling: (n_samples, n_classes) or (n_samples,)
            if decisions.ndim == 1:
                # binary case, map to two-class probability
                decisions = _np.vstack([-decisions, decisions]).T
            probs = softmax(decisions, axis=1)
            classes = model.classes_
            return classes, probs
        except Exception:
            # fallback: deterministic prediction with prob 1
            preds = model.predict(texts)
            classes = model.classes_
            probs = _np.zeros((len(texts), len(classes)))
            for i, p in enumerate(preds):
                idx = list(classes).index(p)
                probs[i, idx] = 1.0
            return classes, probs

# Routing logic (kept simple and configurable)
def route_ticket(category: str, urgency: str) -> str:
    if category == "General/Irrelevant":
        return "Auto-Reply / Bot"
    if category == "Billing & Payments":
        return "Finance Team"
    if category == "Technical Support":
        return "L2 Tech Support" if urgency in ["High", "Critical"] else "L1 Helpdesk"
    if category == "Product Inquiry":
        return "Sales Team"
    if category == "Returns & Refunds":
        return "Logistics & Returns"
    if category == "Account Management":
        return "Customer Success"
    return "General Support"

# Cached data loader
@st.cache_data
def load_dataset():
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# UI Sections
def show_classifier(cat_model, urg_model):
    st.header("🎫 New Ticket Triage")
    col1, col2 = st.columns([2, 1])

    with col1:
        ticket_text = st.text_area("Enter Customer Ticket Text:", height=150, placeholder="e.g., I was charged twice for my subscription...")
        if st.button("Analyze Ticket"):
            if not ticket_text:
                st.warning("Please enter some ticket text.")
                return
            if not cat_model or not urg_model:
                st.error("Models are not loaded.")
                return

            # Predict probabilities and classes
            cat_classes, cat_probs = safe_predict_proba(cat_model, [ticket_text])
            urg_classes, urg_probs = safe_predict_proba(urg_model, [ticket_text])

            cat_idx = int(np.argmax(cat_probs[0]))
            urg_idx = int(np.argmax(urg_probs[0]))
            category = cat_classes[cat_idx]
            urgency = urg_classes[urg_idx]
            cat_conf = float(cat_probs[0][cat_idx])
            urg_conf = float(urg_probs[0][urg_idx])

            CONFIDENCE_THRESHOLD = 0.5
            low_conf = cat_conf < CONFIDENCE_THRESHOLD

            if low_conf or category == "General/Irrelevant":
                category = "General/Irrelevant"
                team = "Auto-Reply / Bot"
                st.warning("⚠️ Flagged as Likely Irrelevant / Low Confidence")
                if low_conf:
                    st.caption(f"Model confidence {cat_conf:.2%} below threshold.")
            else:
                team = route_ticket(category, urgency)

            st.success("Analysis complete")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("**Category**")
                st.metric(label=category, value=f"{cat_conf:.0%}")
            with r2:
                st.markdown("**Urgency**")
                st.metric(label=urgency, value=f"{urg_conf:.0%}")
            with r3:
                st.markdown("**Routed To**")
                st.info(team)

            st.markdown("#### 🤖 Suggested AI Response")
            suggested = generate_response(category, urgency, ticket_text)
            st.text_area("Draft Reply", value=suggested, height=150)

            # Confidence charts
            cat_df = pd.DataFrame({"Category": cat_classes, "Probability": cat_probs[0]})
            urg_df = pd.DataFrame({"Urgency": urg_classes, "Probability": urg_probs[0]})
            c1, c2 = st.columns(2)
            with c1:
                fig_cat = px.bar(cat_df, x="Probability", y="Category", orientation="h", title="Category Scores")
                st.plotly_chart(fig_cat, use_container_width=True)
            with c2:
                fig_urg = px.bar(urg_df, x="Probability", y="Urgency", orientation="h", title="Urgency Scores")
                st.plotly_chart(fig_urg, use_container_width=True)

    with col2:
        st.markdown("### ℹ️ How it works")
        st.info(
            "1. Input ticket text\n"
            "2. Model predicts category & urgency\n"
            "3. System suggests route and draft response\n"
            "4. Provide feedback to improve models"
        )

def show_bulk_classifier(cat_model, urg_model):
    st.header("📂 Bulk Ticket Classification")
    st.markdown("Upload a CSV file with text column to classify multiple tickets.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is None:
        return
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    # Find text column
    text_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ("text", "ticket", "description")):
            text_col = col
            break
    if text_col is None:
        st.error("No suitable text column found. Ensure a column like 'text' or 'description' exists.")
        st.dataframe(df.head())
        return

    if st.button("Run Bulk Classification"):
        if not cat_model or not urg_model:
            st.error("Models are not loaded.")
            return
        with st.spinner("Classifying..."):
            df["Predicted Category"] = cat_model.predict(df[text_col])
            df["Predicted Urgency"] = urg_model.predict(df[text_col])
            df["Routed Team"] = df.apply(lambda r: route_ticket(r["Predicted Category"], r["Predicted Urgency"]), axis=1)
            st.session_state["bulk_df"] = df
            st.success("Done!")

    if "bulk_df" in st.session_state:
        out_df = st.session_state["bulk_df"]
        st.dataframe(out_df.head(), use_container_width=True)
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "classified_tickets.csv", "text/csv")

def show_dashboard():
    st.header("📊 Dashboard")
    df = load_dataset()
    if df.empty:
        st.warning("No local dataset found (`support_tickets_data.csv`). Upload or generate data to see dashboard metrics.")
        return

    total = len(df)
    critical = len(df[df["urgency"] == "Critical"]) if "urgency" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tickets", total)
    c2.metric("Critical Tickets", critical)
    c3.metric("Categories", df["category"].nunique() if "category" in df.columns else 0)

    if "category" in df.columns:
        fig = px.pie(df, names="category", title="Tickets by Category")
        st.plotly_chart(fig, use_container_width=True)

def show_insights(cat_model, urg_model):
    st.header("📈 Model Insights")
    df = load_dataset()
    if df.empty:
        st.warning("No data available for insights.")
        return
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import confusion_matrix

        X = df["text"].fillna("")
        y_cat = df["category"]
        y_urg = df["urgency"]

        if cat_model:
            cat_cv = cross_val_score(cat_model, X, y_cat, cv=5, scoring="accuracy")
            st.metric("Category CV Accuracy", f"{cat_cv.mean():.2%}")
        if urg_model:
            urg_cv = cross_val_score(urg_model, X, y_urg, cv=5, scoring="accuracy")
            st.metric("Urgency CV Accuracy", f"{urg_cv.mean():.2%}")

        # Small train/test confusion matrices if models exist
        X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
            X, y_cat, y_urg, test_size=0.2, random_state=42
        )
        if cat_model:
            y_pred_cat = cat_model.predict(X_test)
            cm_cat = confusion_matrix(y_cat_test, y_pred_cat, labels=sorted(df["category"].unique()))
            fig_cat = px.imshow(cm_cat, labels=dict(x="Predicted", y="Actual", color="Count"), title="Category Confusion")
            st.plotly_chart(fig_cat, use_container_width=True)
        if urg_model:
            y_pred_urg = urg_model.predict(X_test)
            cm_urg = confusion_matrix(y_urg_test, y_pred_urg, labels=sorted(df["urgency"].unique()))
            fig_urg = px.imshow(cm_urg, labels=dict(x="Predicted", y="Actual", color="Count"), title="Urgency Confusion")
            st.plotly_chart(fig_urg, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute insights: {e}")

def generate_response(category, urgency, customer_text):
    responses = {
        "Billing & Payments": "Thank you for contacting billing support. Our finance team is reviewing your transaction.",
        "Technical Support": "We have received your technical support request. Our engineering team is looking into this.",
        "Product Inquiry": "Thanks for your interest. A sales representative will get back to you.",
        "Returns & Refunds": "We're sorry to hear you want to return an item. Our logistics team will follow up.",
        "Account Management": "We can help with account settings. Please verify your account when contacted.",
        "General/Irrelevant": "Your query appears outside our standard support topics. Please check the FAQ.",
        "General Support": "Thank you for reaching out. We will route your query appropriately.",
    }
    base = responses.get(category, responses["General Support"])
    if urgency in ["High", "Critical"] and category != "General/Irrelevant":
        return f"URGENT: {base} We are prioritizing this ticket due to high urgency."
    return f"Hi,\n\n{base}\n\nBest regards,\nSupportFlow Team"

def main():
    apply_dark_theme()
    st.title("🤖 SupportFlow: AI Ticket Classification & Routing")
    st.markdown("---")
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox("", ["Ticket Classifier", "Bulk Classification", "Dashboard", "Model Insights"])

        st.markdown("---")
        st.markdown("Models")
        cat_model, urg_model = load_models()
        if cat_model is None or urg_model is None:
            if st.button("Train Models (local)"):
                ok = train_models_via_script()
                if ok:
                    load_models.clear()
                    st.experimental_rerun()
        else:
            st.success("Models loaded")

    # Ensure models available for pages that need them
    cat_model, urg_model = ensure_models_loaded()

    if page == "Ticket Classifier":
        show_classifier(cat_model, urg_model)
    elif page == "Bulk Classification":
        show_bulk_classifier(cat_model, urg_model)
    elif page == "Dashboard":
        show_dashboard()
    elif page == "Model Insights":
        show_insights(cat_model, urg_model)

if __name__ == "__main__":
    main()