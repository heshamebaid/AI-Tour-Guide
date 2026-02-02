@echo off
REM Delete the existing venv directory
rmdir /s /q venv

REM Recreate the virtual environment
python -m venv venv

REM Activate the new virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo Virtual environment has been reset and dependencies installed. 