@echo off
(
  echo access http://127.0.0.1:8000/docs#/
  echo or input Windows Power Shell 
  echo Invoke-RestMethod -Uri "http://127.0.0.1:8000/search?query=something_keywords"
)
uvicorn main:app --reload