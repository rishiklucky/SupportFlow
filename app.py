import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import numpy as np

# Page Config
st.set_page_config(
    page_title="SupportFlow AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply dark theme as default
def apply_dark_theme():
    st.markdown("""
    <style>
    :root {
        --bg-primary: #111111;
        --bg-secondary: #1e1e1e;
        --text-primary: #ffffff;
        --text-secondary: #e0e0e0;
        --border: #333333;
        --accent: #00d4ff;
    }
    
    body {
        background-color: #111111;
        color: #ffffff;
    }
    
    .stContainer {
        background-color: #111111;
        color: #ffffff;
    }
    
    .stMainBlockContainer {
        background-color: #111111;
        color: #ffffff;
    }
    
    .stSidebar {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
        border-bottom: 2px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    .stDataFrame {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .stMetric {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333333;
    }
    
    .stMetricLabel {
        color: #e0e0e0;
    }
    
    .stMetricValue {
        color: #ffffff;
    }
    
    .stSelectbox, .stMultiSelect, .stTextInput, .stTextArea {
        color: #ffffff;
        background-color: #1e1e1e;
    }
    
    .stButton > button {
        background-color: #00d4ff;
        color: #111111;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #00b8d4;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    .stExpander {
        border-color: #333333;
        color: #ffffff;
    }
    
    .stDivider {
        border-color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        cat_model = joblib.load("models/category_model.pkl")
        urg_model = joblib.load("models/urgency_model.pkl")
        return cat_model, urg_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run train_models.py first.")
        return None, None

cat_model, urg_model = load_models()

# Routing Logic
def route_ticket(category, urgency):
    if category == "General/Irrelevant":
        return "Auto-Reply / Bot"
    elif category == "Billing & Payments":
        return "Finance Team"
    elif category == "Technical Support":
        if urgency in ["High", "Critical"]:
            return "L2 Tech Support"
        else:
            return "L1 Helpdesk"
    elif category == "Product Inquiry":
        return "Sales Team"
    elif category == "Returns & Refunds":
        return "Logistics & Returns"
    elif category == "Account Management":
        return "Customer Success"
    else:
        return "General Support"

def main():
    # Apply dark theme
    apply_dark_theme()
    
    st.title("🤖 SupportFlow: AI Ticket Classification & Routing")
    st.markdown("---")

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        page = st.selectbox("", [ "Ticket Classifier", "Bulk Classification","Dashboard", "Model Insights"], label_visibility="collapsed")

    if page == "Dashboard":
        show_dashboard()
    elif page == "Ticket Classifier":
        show_classifier()
    elif page == "Bulk Classification":
        show_bulk_classifier()
    elif page == "Model Insights":
        show_insights()

def generate_response(category, urgency, customer_text):
    """Simple template-based response generator."""
    responses = {
        "Billing & Payments": "Thank you for contacting billing support. We understand this is regarding a payment issue. Our finance team is reviewing your transaction.",
        "Technical Support": "We have received your technical support request. Our engineering team is looking into the logs.",
        "Product Inquiry": "Thanks for your interest in our products. A sales representative will get back to you with the details requested.",
        "Returns & Refunds": "We are sorry to hear you want to return an item. Our logistics team will process your request shortly.",
        "Account Management": "We can help you manage your account settings. For security reasons, please verify your email when our agent contacts you.",
        "General/Irrelevant": "It seems your query is not related to our standard support topics. Please check our FAQ or rephrase your question.",
        "General Support": "Thank you for reaching out. We will direct your query to the appropriate department."
    }
    
    base_response = responses.get(category, responses["General Support"])
    
    if urgency in ["Critical", "High"] and category != "General/Irrelevant":
        return f"URGENT: {base_response} We are prioritizing this ticket due to its high urgency."
    else:
        return f"Hi there, \n\n{base_response}\n\nBest regards,\nSupportFlow Team"

def show_classifier():
    st.header("🎫 New Ticket Triage")
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticket_text = st.text_area("Enter Customer Ticket Text:", height=150, placeholder="e.g., I was charged twice for my subscription...")
            
            if st.button("Analyze Ticket", type="primary"):
                if ticket_text and cat_model and urg_model:
                    # Confidence Scores
                    cat_probs = cat_model.predict_proba([ticket_text])[0]
                    urg_probs = urg_model.predict_proba([ticket_text])[0]
                    
                    # Get predicted indices
                    cat_idx = np.argmax(cat_probs)
                    urg_idx = np.argmax(urg_probs)
                    
                    # Prediction (Raw)
                    category = cat_model.classes_[cat_idx]
                    urgency = urg_model.classes_[urg_idx]
                    
                    cat_conf = cat_probs[cat_idx]
                    urg_conf = urg_probs[urg_idx]
                    
                    # Threshold for "General/Irrelevant" fallback
                    CONFIDENCE_THRESHOLD = 0.5
                    is_low_conf = cat_conf < CONFIDENCE_THRESHOLD

                    if is_low_conf or category == "General/Irrelevant":
                        # If model is unsure, force it to Irrelevant bucket
                        category = "General/Irrelevant"
                        team = "Auto-Reply / Bot" # Direct override
                        
                        st.warning("⚠️ Flagged as Likely Irrelevant / Low Confidence")
                        if is_low_conf:
                            st.caption(f"Reason: Model confidence ({cat_conf:.2%}) is below threshold.")
                    else:
                        # Routing
                        team = route_ticket(category, urgency)
                    
                    # --- Results Display ---
                    st.success("Analysis Complete!")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.info(f"**Category**\n\n### {category}")
                        st.caption(f"Confidence: {cat_conf:.2%}")
                        
                    with res_col2:
                        color = "red" if urgency in ["High", "Critical"] else "orange" if urgency == "Medium" else "green"
                        st.markdown(f"**Urgency**\n\n### :{color}[{urgency}]")
                        st.caption(f"Confidence: {urg_conf:.2%}")

                    with res_col3:
                        st.warning(f"**Routed To**\n\n### {team}")
                    
                    # Suggested Response
                    st.markdown("#### 🤖 Suggested AI Response")
                    suggested_reply = generate_response(category, urgency, ticket_text)
                    st.text_area("Draft Reply", value=suggested_reply, height=150)

                    # Explainability (Simple feature importance proxy or confidence view)
                    st.markdown("#### 🧠 Model Confidence Breakdown")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        cat_classes = cat_model.classes_
                        cat_df = pd.DataFrame({"Category": cat_classes, "Probability": cat_probs})
                        fig_cat = px.bar(cat_df, x="Probability", y="Category", orientation='h', title="Category Prediction Scores")
                        st.plotly_chart(fig_cat, use_container_width=True)
                        
                    with c2:
                        urg_classes = urg_model.classes_
                        urg_df = pd.DataFrame({"Urgency": urg_classes, "Probability": urg_probs})
                        fig_urg = px.bar(urg_df, x="Probability", y="Urgency", orientation='h', title="Urgency Prediction Scores")
                        st.plotly_chart(fig_urg, use_container_width=True)
                        
                    # Feedback Loop simulation
                    st.markdown("---")
                    with st.expander("Is this correct? Provide Feedback (Active Learning)"):
                        correct = st.checkbox("Prediction is correct", value=True)
                        if not correct:
                            new_cat = st.selectbox("Correct Category", cat_model.classes_)
                            new_urg = st.selectbox("Correct Urgency", urg_model.classes_)
                            if st.button("Submit Feedback"):
                                st.toast("Feedback received! This will be used to retrain the model.")
                elif not ticket_text:
                    st.warning("Please enter some text.")
                else:
                    st.error("Models not loaded.")

        with col2:
            st.markdown("### ℹ️ How it works")
            st.info("""
            1. **Input**: Paste the customer email or chat message.
            2. **Analyze**: The AI scans for keywords and sentiment (simplified).
            3. **Route**: Based on the predicted Category and Urgency, the ticket is assigned to the best team.
            4. **Response**: An auto-generated draft reply is provided.
            """)

def show_bulk_classifier():
    st.header("📂 Bulk Ticket Classification")
    st.markdown("Upload a CSV file with customer ID and text columns to classify multiple tickets at once.")
    
    uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")
    
    if uploaded_file and cat_model and urg_model:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find customer ID column
            customer_id_col = None
            for col in df.columns:
                if "customer" in col.lower() or "id" in col.lower() or "customer_id" in col.lower():
                    customer_id_col = col
                    break
            
            # Find text column
            text_col = None
            for col in df.columns:
                if "text" in col.lower() or "ticket" in col.lower() or "description" in col.lower():
                    text_col = col
                    break
            
            if text_col:
                st.info(f"Using column `{text_col}` for classification" + (f" and `{customer_id_col}` for customer IDs." if customer_id_col else "."))
                
                if st.button("Run Bulk Classification"):
                    with st.spinner("Classifying tickets..."):
                        # Predict
                        df["Predicted Category"] = cat_model.predict(df[text_col])
                        df["Predicted Urgency"] = urg_model.predict(df[text_col])
                        df["Routed Team"] = df.apply(lambda x: route_ticket(x["Predicted Category"], x["Predicted Urgency"]), axis=1)
                        
                        # Store in session state for filtering
                        st.session_state.classified_df = df
                        st.session_state.customer_id_col = customer_id_col
                        st.session_state.text_col = text_col
                        
                        st.success("Classification complete!")
                
                # Display and filter results
                if "classified_df" in st.session_state:
                    df = st.session_state.classified_df
                    
                    st.markdown("---")
                    st.subheader("📊 Classification Results & Analytics")
                    
                    # Summary Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Tickets", len(df))
                    m2.metric("Categories", df["Predicted Category"].nunique())
                    m3.metric("Urgency Levels", df["Predicted Urgency"].nunique())
                    m4.metric("Routed Teams", df["Routed Team"].nunique())
                    
                    # Visualization Dashboard
                    st.markdown("---")
                    st.markdown("### 📈 Classification Dashboard")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Category Distribution
                        cat_counts = df["Predicted Category"].value_counts().reset_index()
                        cat_counts.columns = ["Category", "Count"]
                        fig_cat = px.bar(cat_counts, x="Category", y="Count", title="Tickets by Category", 
                                        color="Count", color_continuous_scale="Blues")
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    with viz_col2:
                        # Urgency Distribution
                        urg_counts = df["Predicted Urgency"].value_counts().reset_index()
                        urg_counts.columns = ["Urgency", "Count"]
                        fig_urg = px.pie(urg_counts, names="Urgency", values="Count", title="Tickets by Urgency Level",
                                        color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig_urg, use_container_width=True)
                    
                    viz_col3, viz_col4 = st.columns(2)
                    
                    with viz_col3:
                        # Team Assignment Distribution
                        team_counts = df["Routed Team"].value_counts().reset_index()
                        team_counts.columns = ["Team", "Count"]
                        fig_team = px.bar(team_counts, y="Team", x="Count", orientation="h", title="Tickets by Routed Team",
                                         color="Count", color_continuous_scale="Viridis")
                        st.plotly_chart(fig_team, use_container_width=True)
                    
                    with viz_col4:
                        # Category vs Urgency Heatmap
                        heatmap_data = pd.crosstab(df["Predicted Category"], df["Predicted Urgency"])
                        fig_heat = px.imshow(heatmap_data, labels=dict(x="Urgency", y="Category", color="Count"),
                                            title="Category vs Urgency Matrix", color_continuous_scale="YlOrRd")
                        st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Full Results Table
                    st.markdown("---")
                    st.markdown("### 📋 All Classifications")
                    display_cols = []
                    if st.session_state.customer_id_col:
                        display_cols.append(st.session_state.customer_id_col)
                    display_cols.extend([st.session_state.text_col, "Predicted Category", "Predicted Urgency", "Routed Team"])
                    
                    st.dataframe(df[display_cols], use_container_width=True)
                    
                    # Interactive filtering
                    st.markdown("---")
                    st.subheader("🔍 Filter by Category or Urgency")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        categories = df["Predicted Category"].unique().tolist()
                        selected_category = st.selectbox("Select Category", ["All"] + sorted(categories))
                        
                        if selected_category != "All":
                            filtered_df = df[df["Predicted Category"] == selected_category]
                            st.markdown(f"#### 👥 Customers in '{selected_category}'")
                            
                            if st.session_state.customer_id_col:
                                customer_ids = filtered_df[st.session_state.customer_id_col].tolist()
                                st.markdown(f"**Total: {len(customer_ids)} customers**")
                                st.dataframe(filtered_df[display_cols], use_container_width=True)
                                
                                # Show customer IDs in a more readable format
                                with st.expander(f"View all {len(customer_ids)} Customer IDs"):
                                    customer_list = "\n".join([f"• {cid}" for cid in customer_ids])
                                    st.text(customer_list)
                            else:
                                st.dataframe(filtered_df[display_cols], use_container_width=True)
                    
                    with col2:
                        urgencies = df["Predicted Urgency"].unique().tolist()
                        selected_urgency = st.selectbox("Select Urgency", ["All"] + sorted(urgencies))
                        
                        if selected_urgency != "All":
                            filtered_df = df[df["Predicted Urgency"] == selected_urgency]
                            st.markdown(f"#### 👥 Customers with '{selected_urgency}' Urgency")
                            
                            if st.session_state.customer_id_col:
                                customer_ids = filtered_df[st.session_state.customer_id_col].tolist()
                                st.markdown(f"**Total: {len(customer_ids)} customers**")
                                st.dataframe(filtered_df[display_cols], use_container_width=True)
                                
                                # Show customer IDs in a more readable format
                                with st.expander(f"View all {len(customer_ids)} Customer IDs"):
                                    customer_list = "\n".join([f"• {cid}" for cid in customer_ids])
                                    st.text(customer_list)
                            else:
                                st.dataframe(filtered_df[display_cols], use_container_width=True)
                    
                    # Download
                    st.markdown("---")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "classified_tickets.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.error("Could not find a suitable text column. Please ensure your CSV has a column like 'text', 'ticket', 'description'.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error processing file: {e}")


def show_dashboard():
    st.header("📊 Live Operations Dashboard")
    
    # Load base data
    try:
        df = pd.read_csv("support_tickets_data.csv")
    except:
        df = pd.DataFrame()

    # 📤 Upload External Data
    with st.expander("Import External Data (CSV)"):
        uploaded_file = st.file_uploader("Upload CSV to merge", type="csv")
        if uploaded_file is not None:
            try:
                ext_df = pd.read_csv(uploaded_file)
                if not df.empty:
                    df = pd.concat([df, ext_df], ignore_index=True)
                else:
                    df = ext_df
                st.success(f"Merged {len(ext_df)} imported records.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if df.empty:
        st.warning("No data found. Please ensure 'support_tickets_data.csv' exists or upload a file.")
        return

    # metrics
    total_tickets = len(df)
    critical_tickets = len(df[df['urgency'] == 'Critical'])
    avg_conf = 0.88 # Simulated metadata

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets Processed", total_tickets, "+12%")
    m2.metric("Critical Issues", critical_tickets, "-2%", delta_color="inverse")
    m3.metric("Avg. Model Confidence", f"{avg_conf:.0%}", "+1%")
    m4.metric("SLA Breaches", "5", "-1", delta_color="inverse")

    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Ticket Volume by Category")
        fig1 = px.pie(df, names='category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Urgency Distribution")
        fig2 = px.bar(df, x='category', color='urgency', title="Urgency by Category", barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    # Keyword Analysis
    st.markdown("---")
    st.subheader("🔑 Topic Analysis (Top Keywords)")
    
    if "text" in df.columns:
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Simple stop words list to avoid dependency on heavy nltk download if not present
        stop_words = frozenset(["the", "a", "an", "in", "on", "of", "to", "for", "is", "im", "am", "i", "my", "me", "please", "help", "issue", "problem", "regarding", "check", "need", "assistance", "with", "trouble", "facing", "having", "question"])
        
        try:
            vec = CountVectorizer(stop_words=list(stop_words), max_features=20)
            X = vec.fit_transform(df['text'].dropna())
            sum_words = X.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
            
            common_words_df = pd.DataFrame(words_freq, columns = ['Word', 'Count'])
            
            fig3 = px.bar(common_words_df, x='Count', y='Word', orientation='h', title="Top Recurring Terms in Tickets")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Could not analyze keywords: {e}")
    else:
        st.info("No 'text' column found for keyword analysis.")

def show_insights():
    st.header("📈 Model Performance & Insights")
    st.markdown("Real-time evaluation metrics using 5-Fold Cross-Validation.")
    
    try:
        # Load data
        df = pd.read_csv("support_tickets_data.csv")
        df['text'] = df['text'].fillna('')
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        X = df['text']
        y_category = df['category']
        y_urgency = df['urgency']
        
        # Load models
        category_model = joblib.load("models/category_model.pkl")
        urgency_model = joblib.load("models/urgency_model.pkl")
        
        # Use 5-fold cross-validation for metrics
        st.markdown("### Classification Accuracy (5-Fold Cross-Validation)")
        
        col1, col2 = st.columns(2)
        
        # Category model cross-validation
        cat_cv_scores = cross_val_score(category_model, X, y_category, cv=5, scoring='accuracy')
        cat_mean_accuracy = cat_cv_scores.mean() * 100
        cat_std_accuracy = cat_cv_scores.std() * 100
        
        with col1:
            st.metric("Category Model Accuracy", 
                     f"{cat_mean_accuracy:.2f}%", 
                     delta=f"±{cat_std_accuracy:.2f}%",
                     delta_color="off")
        
        # Urgency model cross-validation
        urg_cv_scores = cross_val_score(urgency_model, X, y_urgency, cv=5, scoring='accuracy')
        urg_mean_accuracy = urg_cv_scores.mean() * 100
        urg_std_accuracy = urg_cv_scores.std() * 100
        
        with col2:
            st.metric("Urgency Model Accuracy", 
                     f"{urg_mean_accuracy:.2f}%", 
                     delta=f"±{urg_std_accuracy:.2f}%",
                     delta_color="off")
        
        # Get test set for confusion matrices
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
            X, y_category, y_urgency, test_size=0.2, random_state=42
        )
        
        category_pred = category_model.predict(X_test)
        urgency_pred = urgency_model.predict(X_test)
        
        # Display confusion matrices
        st.markdown("### Model Confusion Matrices")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Model")
            cat_cm = confusion_matrix(y_cat_test, category_pred)
            categories = sorted(df['category'].unique())
            fig_cat = px.imshow(cat_cm, labels=dict(x="Predicted", y="Actual", color="Count"), 
                               x=categories, y=categories, color_continuous_scale='Blues',
                               title="Category Confusion Matrix")
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            st.subheader("Urgency Model")
            urg_cm = confusion_matrix(y_urg_test, urgency_pred)
            urgencies = sorted(df['urgency'].unique())
            fig_urg = px.imshow(urg_cm, labels=dict(x="Predicted", y="Actual", color="Count"), 
                               x=urgencies, y=urgencies, color_continuous_scale='Greens',
                               title="Urgency Confusion Matrix")
            st.plotly_chart(fig_urg, use_container_width=True)
    
    except FileNotFoundError as e:
        st.warning(f"⚠️ Required files not found: {e}")
        st.info("Please ensure you have run `train_models.py` to generate the models and have the data file in place.")

if __name__ == "__main__":
    main()
