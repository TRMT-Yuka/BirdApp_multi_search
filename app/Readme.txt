【API起動】
cd app
uvicorn main:app --reload

or

start_app実行

【実行】
Windows Power Shellからクエリ送信
	curl "http://127.0.0.1:8000/ja_search?query=ヤマゲラ"
	Invoke-RestMethod -Uri "http://127.0.0.1:8000/ja_search?query=ヤマゲラ"


GUI
http://127.0.0.1:8000/docs#/
へアクセス

自作GUI

http://127.0.0.1:8000/word_search
http://127.0.0.1:8000/sound_search