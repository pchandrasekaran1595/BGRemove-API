start /MAX cmd /c "title API-SERVER && cd venv/Scripts && activate && cd .. && cd .. && cls && uvicorn main:app --port 3030 --reload"