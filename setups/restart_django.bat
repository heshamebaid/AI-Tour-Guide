@echo off
echo Clearing Python cache...
del /s /q "Agentic_RAG\src\__pycache__" 2>nul
del /s /q "Agentic_RAG\src\services\__pycache__" 2>nul
del /s /q "Django\myapp\__pycache__" 2>nul

echo.
echo Restarting Django server...
cd Django
call conda activate EduMentorAI
python manage.py runserver
