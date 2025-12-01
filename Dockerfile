# 1. 베이스 이미지를 명확한 버전(Debian Bookworm)으로 고정하여 안정성 확보
FROM python:3.9-slim-bookworm

# 2. apt-get update 실패 방지를 위한 설정 및 패키지 설치
# libgl1-mesa-glx 대신 libgl1 사용 (최신 버전 호환)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
