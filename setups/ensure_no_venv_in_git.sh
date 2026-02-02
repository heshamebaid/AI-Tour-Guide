#!/usr/bin/env bash
# Run once if venv was ever committed - removes from index only (keeps files on disk).
# After running: commit the change so future pushes don't include venv.
set -e
echo "Removing any tracked venv/ from git index..."
git rm -r --cached Chatbot/venv 2>/dev/null || true
git rm -r --cached Django/venv 2>/dev/null || true
git rm -r --cached venv 2>/dev/null || true
git rm -r --cached .venv 2>/dev/null || true
find . -type d -name venv -o -type d -name .venv 2>/dev/null | while read -r d; do
  git rm -r --cached "$d" 2>/dev/null || true
done
echo "Done. Check: git status"
