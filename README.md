# SupportFlow : AI-Driven Customer Support Ticket Classification and Routing System

## Project Overview
This project is an AI-powered system designed to automate the classification and routing of customer support tickets. It utilizes Natural Language Processing (NLP) and Machine Learning (ML) to analyze ticket content, predict the issue category and urgency, and route it to the appropriate support team.

## Features
- **Ticket Classification**: Automatically categorizes tickets (e.g., Billing, Technical Support, Returns).
- **Urgency Prediction**: Predicts the urgency level (Low, Medium, High, Critical).
- **Automated Routing**: Assigns tickets to the correct team based on category and urgency.
- **Interactive Dashboard**: A Streamlit-based UI for agents to view, analyze, and process tickets.
- **Analytics**: Visualization of ticket trends and model performance.

## Tech Stack
- **Language**: Python
- **UI Framework**: Streamlit
- **ML/NLP**: Scikit-learn, Pandas, Numpy
- **Visualization**: Plotly

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate synthetic data (if no data exists):
   ```bash
   python generate_data.py
   ```
3. Train the models:
   ```bash
   python train_models.py
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
##
   You Can Access the Website from here : https://supportflow1.streamlit.app/
