@echo off
REM Run once if venv was ever committed - removes from index only (keeps files on disk).
REM After running: commit the change so future pushes don't include venv.
echo Removing any tracked venv/ from git index...
git rm -r --cached Chatbot/venv 2>nul
git rm -r --cached Django/venv 2>nul
git rm -r --cached venv 2>nul
git rm -r --cached "**/venv" 2>nul
for /d /r . %%d in (venv) do git rm -r --cached "%%d" 2>nul
echo Done. Check: git status
pause
