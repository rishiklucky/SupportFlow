# SupportFlow AI - Intelligent Customer Support Ticket System

![SupportFlow](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

##  Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Customization](#customization)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [Learning Resources](#learning-resources)
- [License](#license)

##  Project Overview

SupportFlow AI is an intelligent customer support ticket classification and routing system powered by Machine Learning. It automatically analyzes incoming support tickets, predicts their category, determines urgency levels, and routes them to the appropriate teams for faster resolution.

This system dramatically reduces manual ticket processing time, improves response efficiency, and ensures tickets reach the right teams immediately.

##  Features

- ** Automatic Classification**: Intelligently categorizes tickets into predefined categories:
  - Billing Issues
  - Technical Support
  - Product Returns
  - General Inquiries
  - Account Management

- ** Urgency Detection**: Predicts ticket priority levels:
  - Low
  - Medium
  - High
  - Critical

- ** Smart Routing**: Automatically assigns tickets to appropriate teams based on category and urgency

- ** Interactive Dashboard**: User-friendly Streamlit interface for:
  - Viewing and processing tickets
  - Real-time analytics and visualizations
  - Model performance metrics
  - Ticket trend analysis

- ** Analytics & Insights**: Comprehensive visualizations showing:
  - Ticket distribution by category
  - Urgency level trends
  - Team workload analysis
  - Processing time statistics

- ** Model Persistence**: Pre-trained models saved for quick inference without retraining

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.8+ |
| **Frontend** | Streamlit |
| **ML/NLP** | Scikit-learn, TF-IDF Vectorizer |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Model Serialization** | Joblib |

##  Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

##  Installation

### Step 1: Clone the Repository
\\\bash
git clone https://github.com/YOUR_USERNAME/SupportFlow.git
cd SupportFlow
\\\

### Step 2: Create a Virtual Environment (Recommended)
\\\bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
\\\

### Step 3: Install Dependencies
\\\bash
pip install -r requirements.txt
\\\

### Step 4: Generate Sample Data (First Time Only)
If you're setting up for the first time and don't have training data:
\\\bash
python generate_data.py
\\\
This creates \support_tickets_data.csv\ with 1000 synthetic support tickets for training.

### Step 5: Train the Models
\\\bash
python train_models.py
\\\
This trains two ML models:
- **Category Model**: Classifies ticket category (Logistic Regression)
- **Urgency Model**: Predicts urgency level (Gradient Boosting)

The trained models are saved in the \models/\ directory.

### Step 6: Run the Application
\\\bash
streamlit run app.py
\\\
The application will open in your default browser at \http://localhost:8501\

**Quick Start Batch File (Windows):**
\\\bash
run_app.bat
\\\

##  Project Structure

\\\
SupportFlow/
 app.py                      # Main Streamlit application
 train_models.py             # Model training script
 generate_data.py            # Synthetic data generation
 support_tickets_data.csv    # Training dataset
 requirements.txt            # Python dependencies
 README.md                   # This file
 run_app.bat                 # Windows batch file to run app
 models/                     # Trained ML models (generated)
    category_model.pkl      # Category classification model
    urgency_model.pkl       # Urgency prediction model
 __pycache__/               # Python cache files
\\\

##  Usage Guide

### 1. **Dashboard Tab**
   - View summary statistics
   - See ticket distribution metrics
   - Monitor system performance
   - Check team workload

### 2. **Process Tickets Tab**
   - Enter new ticket information
   - View AI predictions
   - Review suggested routing
   - Manual override options if needed

### 3. **Analytics Tab**
   - Visual breakdown of tickets by category
   - Urgency level distributions
   - Performance metrics
   - Historical trends

### 4. **Bulk Upload Tab**
   - Upload CSV files with multiple tickets
   - Batch process tickets
   - Download results

##  How It Works

### Data Flow

\\\
Input Ticket
      |
      v
Text Preprocessing (Cleaning, Lowercasing, etc.)
      |
      v   
TF-IDF Vectorization (Convert text to numerical features)
      |
      v   
Category Model (Logistic Regression)
      |
      v   
Urgency Model (Gradient Boosting)
      |
      v    
Routing Decision
      |
      v
Output: Category, Urgency, Assigned Team
\\\

### Feature Engineering
- **TF-IDF Vectorization**: Converts ticket text into numerical features
- **Bi-gram Support**: Captures two-word phrases for better context
- **Stop Words Removal**: Filters common English words
- **Max Features**: Limited to 3000 most important words
- **Min/Max Document Frequency**: Filters out rare and overly common terms

##  Model Details

### Category Model
- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF with bi-grams
- **Input**: Ticket text
- **Output**: Category (Billing, Technical, Returns, etc.)
- **Training Data**: 1000+ synthetic tickets

### Urgency Model
- **Algorithm**: Gradient Boosting Classifier
- **Parameters**: 
  - n_estimators: 150
  - learning_rate: 0.05
  - max_depth: 5
- **Input**: Ticket text
- **Output**: Urgency Level (Low, Medium, High, Critical)
- **Training Data**: 1000+ synthetic tickets

##  Customization

### Add New Categories
1. Modify \generate_data.py\ to include new categories
2. Regenerate data with \python generate_data.py\
3. Retrain models with \python train_models.py\
4. Update routing logic in \app.py\

### Adjust Model Parameters
Edit the hyperparameters in \	rain_models.py\:
- Change TF-IDF parameters (max_features, ngram_range, etc.)
- Adjust Gradient Boosting settings
- Modify validation split ratio

### Use Your Own Data
Replace \support_tickets_data.csv\ with your dataset ensuring it has columns:
- \	ext\: Ticket content
- \category\: Ticket category
- \urgency\: Urgency level

##  Performance Metrics

The system displays:
- **Accuracy Score**: Overall model accuracy on test data
- **Processing Time**: Average time to classify a ticket
- **Ticket Throughput**: Tickets processed per minute
- **Category-wise Accuracy**: Performance per category
- **Urgency Prediction Accuracy**: Precision for each urgency level

##  Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (\git checkout -b feature/AmazingFeature\)
3. Commit your changes (\git commit -m 'Add some AmazingFeature'\)
4. Push to the branch (\git push origin feature/AmazingFeature\)
5. Open a Pull Request

##  Troubleshooting

### Models not found error
**Solution**: Run \python train_models.py\ to train and save the models.

### Data file not found error
**Solution**: Run \python generate_data.py\ to create sample data.

### Port 8501 already in use
**Solution**: Run \streamlit run app.py --server.port 8502\

### High memory usage
**Solution**: Reduce \max_features\ in TF-IDF vectorizer in \	rain_models.py\



