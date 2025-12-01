# 1. 파이썬 환경 이미지를 베이스로 사용
FROM python:3.9-slim

# 2. 필수 시스템 패키지 설치 (Tesseract OCR 포함)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행 (Gunicorn 사용)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
