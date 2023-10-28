# ベースイメージを選択
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# FastAPIアプリケーションのコードをコピー
COPY app /app