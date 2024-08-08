# ベースイメージとしてPythonを使用
FROM python:3.9-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    wget \
    && apt-get clean

# 作業ディレクトリを作成
WORKDIR /app

# 必要なPythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 認証情報をコンテナにコピー
COPY key/lispo-stg-b207dd3f573e.json /app/keyfile.json

# Google Cloud Vision APIの認証情報環境変数を設定
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"

# ラベル、イメージ情報をコピー
COPY labels/sample-image3.xml /app/labels/sample-image3.xml
COPY images /app/images
# ソースコードをコピー
COPY main.py /app/main.py

# コンテナを起動した際に実行されるコマンドを指定
CMD ["python", "main.py"]
