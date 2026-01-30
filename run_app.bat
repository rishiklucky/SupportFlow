@echo off
echo Starting SupportFlow AI...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
if not exist "models\category_model.pkl" (
    echo Generating data and training models...
    python generate_data.py
    python train_models.py
)
echo Launching Application...
streamlit run app.py
pause
