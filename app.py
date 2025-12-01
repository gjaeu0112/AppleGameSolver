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
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 (White vs Black 구분)
    # 배경(255)과 글씨(255)는 흰색, 사과(0)는 검은색이 되도록 강하게 나눕니다.
    # 180~200 이상을 흰색으로 잡습니다.
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 4. [핵심] 배경 지우기 (Flood Fill)
    # 이미지의 (0,0) 좌표는 무조건 배경(흰색)이라고 가정하고,
    # 여기서부터 연결된 모든 흰색을 검은색(0)으로 칠해버립니다.
    # 사과(검은색)가 벽 역할을 해서, 사과 속에 있는 글씨(흰색)에는 페인트가 닿지 않습니다.
    
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # 배경 제거용 복사본 생성
    flooded = binary.copy()
    
    # (0,0)에서 시작해 연결된 흰색을 검은색으로 채움
    cv2.floodFill(flooded, mask, (0, 0), 0)
    
    # 만약 테두리가 잘려서 (0,0)이 사과일 수도 있으니, 네 귀퉁이를 다 시도합니다.
    cv2.floodFill(flooded, mask, (w-1, 0), 0)
    cv2.floodFill(flooded, mask, (0, h-1), 0)
    cv2.floodFill(flooded, mask, (w-1, h-1), 0)
    
    # 이제 'flooded' 이미지에는 "사과 속의 흰 글씨"만 흰색으로 남고 나머지는 다 검은색입니다.
    
    # 5. 색상 반전
    # Tesseract는 "흰 배경에 검은 글씨"를 좋아하므로 반전시킵니다.
    final_img = cv2.bitwise_not(flooded)
    
    # 6. Tesseract 실행
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(final_img, config=config)
    
    # 7. 숫자 추출 및 결과 정리
    all_digits = [int(char) for char in text if char.isdigit()]
    
    ROWS, COLS = 10, 17
    target_count = ROWS * COLS
    
    print(f"🔎 찾은 숫자: {len(all_digits)}개")
    
    if len(all_digits) < target_count:
        all_digits += [0] * (target_count - len(all_digits))
    elif len(all_digits) > target_count:
        all_digits = all_digits[:target_count]
        
    board = []
    for r in range(ROWS):
        board.append(all_digits[r*COLS : (r+1)*COLS])
        
    # 미리보기 이미지 생성 (제대로 배경이 지워졌는지 확인용)
    preview_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
    
    return board, preview_img
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
