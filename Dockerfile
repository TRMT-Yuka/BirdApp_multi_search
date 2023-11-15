# ベースイメージを選択
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# FastAPIアプリケーションのコードをコピー
COPY app /app

# Expose port 8000 for FastAPI to run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]