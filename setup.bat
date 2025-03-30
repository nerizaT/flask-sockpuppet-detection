@echo off
cd /d %~dp0
echo [INFO] Setting up virtual environment...

if exist venv rmdir /s /q venv
python -m venv venv
call venv\Scripts\activate

echo [INFO] Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
pip install Flask pandas numpy textblob joblib scikit-learn sentence-transformers

echo [SUCCESS] Setup complete!
pause