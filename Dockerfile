# ベースイメージを選択
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 作業ディレクトリを設定
WORKDIR /my_codes

# Copy the current directory contents into the container at /app
COPY . /my_codes

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod -R 777 /my_codes/app

WORKDIR /my_codes/app

RUN mkdir CACHE && chmod -R 777 CACHE
ENV TRANSFORMERS_CACHE CACHE
ENV NUMBA_CACHE_DIR CACHE

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
