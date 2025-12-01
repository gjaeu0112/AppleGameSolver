import sys
import os
import cv2
import numpy as np
import pytesseract
import base64
from flask import Flask, request, jsonify, render_template_string

# ==========================================
# 1. 설정 및 초기화
# ==========================================
app = Flask(__name__)
sys.setrecursionlimit(20000)
# ==========================================
# 2. HTML 프론트엔드 (편의상 여기에 포함)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>사과 게임 솔버 🍎</title>
    <style>
        body { font-family: 'Apple SD Gothic Neo', sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #e84118; margin-bottom: 10px; }
        .desc { color: #666; margin-bottom: 30px; font-size: 14px; }
        
        .upload-box { border: 2px dashed #ccc; padding: 30px; border-radius: 10px; cursor: pointer; transition: 0.3s; }
        .upload-box:hover { border-color: #e84118; background: #fff0f0; }
        
        button { background: #e84118; color: white; border: none; padding: 12px 30px; border-radius: 25px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 20px; width: 100%; transition: 0.2s; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        
        #preview, #result { max-width: 100%; margin-top: 20px; border-radius: 8px; display: none; }
        .loader { display: none; margin: 20px auto; border: 4px solid #f3f3f3; border-top: 4px solid #e84118; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .score-box { background: #2f3640; color: #fbc531; padding: 15px; border-radius: 8px; margin-top: 20px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍎 사과 게임 솔버</h1>
        <p class="desc">게임 스크린샷을 올리면 AI가 10초 안에 만점을 찾아줍니다.</p>
        
        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            <p>📷 이미지를 클릭하여 업로드</p>
            <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="previewImage()">
        </div>
        <img id="preview" src="">
        
        <button id="solveBtn" onclick="uploadAndSolve()">분석 시작</button>
        
        <div class="loader" id="loader"></div>
        
        <div class="score-box" id="scoreDisplay"></div>
        <img id="result" src="">
    </div>

    <script>
        function previewImage() {
            const file = document.getElementById('fileInput').files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById('preview');
                    img.src = e.target.result;
                    img.style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                    document.getElementById('scoreDisplay').style.display = 'none';
                }
                reader.readAsDataURL(file);
            }
        }

        async function uploadAndSolve() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) { alert("이미지를 먼저 선택해주세요."); return; }

            const btn = document.getElementById('solveBtn');
            const loader = document.getElementById('loader');
            const resultImg = document.getElementById('result');
            const scoreDisplay = document.getElementById('scoreDisplay');

            btn.disabled = true;
            btn.innerText = "분석 중...";
            loader.style.display = 'block';
            resultImg.style.display = 'none';
            scoreDisplay.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/solve', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.error) {
                    alert("에러 발생: " + data.error);
                } else {
                    scoreDisplay.innerText = "🏆 예상 최고 점수: " + data.score + "점";
                    scoreDisplay.style.display = 'block';
                    resultImg.src = "data:image/jpeg;base64," + data.image;
                    resultImg.style.display = 'block';
                }
            } catch (e) {
                alert("서버 연결 실패");
            } finally {
                btn.disabled = false;
                btn.innerText = "분석 시작";
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# ==========================================
# 3. OCR 및 이미지 처리 로직
# ==========================================
import re

def extract_grid_from_image(img_stream):
    # 1. 이미지 읽기
    file_bytes = np.frombuffer(img_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 2. 최소 전처리: "배경만 지운다"
    # 사과 그림(빨강/초록) 때문에 Tesseract가 헷갈릴 수 있으므로,
    # 흑백으로 바꾸고 대비를 극대화(Threshold)하여 '흰 글씨'만 남깁니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu 알고리즘: 배경과 글씨를 나누는 최적의 값을 자동으로 찾음
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 노이즈 제거 (점 같은 것 없애기) - 선택 사항
    # binary = cv2.medianBlur(binary, 3) 

    # 3. Tesseract에 통째로 전송
    # --psm 6: 이미지를 하나의 균일한 텍스트 뭉치(Block)로 취급
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(binary, config=config)
    
    # 4. 결과 파싱 (텍스트 -> 리스트)
    # 공백, 줄바꿈 다 무시하고 오직 '숫자'만 싹 긁어모음
    all_digits = [int(char) for char in text if char.isdigit()]
    
    ROWS, COLS = 10, 17
    target_count = ROWS * COLS # 170개
    
    print(f"인식된 숫자 개수: {len(all_digits)} / {target_count}")
    
    # [보정 로직] 개수가 안 맞을 경우
    if len(all_digits) < target_count:
        # 부족하면 뒤를 0으로 채움 (최소한 에러는 안 나게)
        all_digits += [0] * (target_count - len(all_digits))
    elif len(all_digits) > target_count:
        # 넘치면(노이즈 인식) 앞에서부터 170개만 자름
        all_digits = all_digits[:target_count]
    
    # 1차원 리스트 -> 10x17 2차원 리스트로 변환
    board = []
    for r in range(ROWS):
        start = r * COLS
        end = (r + 1) * COLS
        board.append(all_digits[start:end])
        
    # 결과 확인용으로 'binary' 이미지를 리턴해서 웹에서 인식 상태를 볼 수 있게 함
    # (제대로 흑백 분리가 되었는지 확인하는 용도)
    processed_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return board, processed_preview
# ==========================================
# 4. 알고리즘 로직 (그래프 기반)
# ==========================================
def solve_puzzle(board):
    R, C = 10, 17
    
    # 1. 유효한 직사각형 탐색
    rects = []
    for r in range(R):
        for c in range(C):
            for h in range(1, R - r + 1):
                for w in range(1, C - c + 1):
                    s = 0
                    for i in range(h):
                        s += sum(board[r+i][c:c+w])
                    
                    if s == 10:
                        score = h * w
                        mask = 0
                        for i in range(h):
                            for j in range(w):
                                mask |= (1 << ((r + i) * C + (c + j)))
                        rects.append({'id': len(rects), 'score': score, 'mask': mask, 'info': (r, c, h, w)})
                    elif s > 10:
                        break
                        
    N = len(rects)
    rects.sort(key=lambda x: x['score'], reverse=True)
    
    # 2. 충돌 그래프 생성
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rects[i]['mask'] & rects[j]['mask']:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. 최대 점수 탐색 (Branch and Bound)
    global_max = 0
    best_solution = []

    def search(current_score, candidates, path):
        nonlocal global_max, best_solution
        
        potential = current_score + sum(rects[idx]['score'] for idx in candidates)
        if potential <= global_max:
            return

        if not candidates:
            if current_score > global_max:
                global_max = current_score
                best_solution = list(path)
            return

        first = candidates[0]
        remaining = candidates[1:]
        
        # Include
        next_candidates = [x for x in remaining if x not in adj[first]]
        path.append(first)
        search(current_score + rects[first]['score'], next_candidates, path)
        path.pop()
        
        # Exclude
        search(current_score, remaining, path)

    search(0, list(range(N)), [])
    
    # 결과 복원
    final_rects = [rects[idx]['info'] for idx in best_solution]
    return global_max, final_rects

# ==========================================
# 5. Flask 라우트
# ==========================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/solve', methods=['POST'])
def solve():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'})
    
    try:
        # 1. OCR
        board, cropped_img = extract_grid_from_image(request.files['file'])
        
        # 2. 알고리즘 풀이
        score, rects = solve_puzzle(board)
        
        # 3. 결과 그리기 (크롭된 이미지 위에)
        result_img = cropped_img.copy()
        h_img, w_img, _ = result_img.shape
        cell_h = h_img // 10
        cell_w = w_img // 17
        
        for (r, c, h, w) in rects:
            cv2.rectangle(result_img, 
                          (c * cell_w, r * cell_h), 
                          ((c + w) * cell_w, (r + h) * cell_h), 
                          (0, 0, 255), 3) # 빨간색 테두리
                          
        # 4. 이미지 인코딩
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'score': score, 'image': img_base64})
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 로컬 테스트 시
    app.run(debug=True, host='0.0.0.0', port=5000)
