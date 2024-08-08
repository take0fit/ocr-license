# PythonイメージをベースにしたDockerfile
FROM python:3.8-slim

# 作業ディレクトリを作成
WORKDIR /app

# 必要な依存関係をインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-jpn \
    wget \
    && apt-get clean

# 日本語データをダウンロード
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

# 必要なファイルをコンテナにコピー
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY images images/

# パッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# Tesseractのデータディレクトリを設定
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# スクリプトを実行するコマンド
CMD ["python3", "main.py", "images/sample-image3.png"]
