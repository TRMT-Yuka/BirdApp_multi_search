API起動

cd app
uvicorn main:app --reload

start_app実行

Windows Power Shellからクエリ送信

curl "http://127.0.0.1:8000/search?query=ヤマゲラ"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/search?query=ヤマゲラ"


http://127.0.0.1:8000/docs#/
へアクセス