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
def extract_grid_from_image(img_stream):
    # ==========================================
    # 1. 이미지 전처리 (공통)
    # ==========================================
    file_bytes = np.frombuffer(img_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("숫자 영역을 찾을 수 없습니다.")
    
    x, y, w, h = cv2.boundingRect(points)
    cropped_img = img[y:y+h, x:x+w] # 결과 시각화용
    processed_area = binary[y:y+h, x:x+w] # 인식용
    
    ROWS, COLS = 10, 17
    cell_h = h // ROWS
    cell_w = w // COLS
    
    # Tesseract 설정
    config_block = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789' # 뭉텅이 인식용
    config_line  = r'--oem 3 --psm 7 -c tessedit_char_whitelist=123456789' # 한 줄 인식용
    config_char  = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789' # 한 글자 인식용

    # ==========================================
    # 2. [1단계] Fast Path: 전체 통으로 읽기 (가장 빠름)
    # ==========================================
    print("Attempt 1: One-shot scan...", end=" ")
    
    canvas_h = ROWS * 40 
    canvas_w = COLS * 30
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    
    # 캔버스에 셀 옮겨심기 (재조립)
    cells_map = [[None for _ in range(COLS)] for _ in range(ROWS)]
    
    for r in range(ROWS):
        for c in range(COLS):
            cy, cx = r * cell_h, c * cell_w
            margin_y, margin_x = int(cell_h * 0.15), int(cell_w * 0.15)
            cell = processed_area[cy+margin_y : cy+cell_h-margin_y, cx+margin_x : cx+cell_w-margin_x]
            cell = cv2.resize(cell, (20, 28))
            cell = cv2.bitwise_not(cell) # 반전 (검은 글씨)
            
            # 나중에 재사용하기 위해 저장
            cells_map[r][c] = cell
            
            # 캔버스에 부착
            target_y, target_x = r * 40 + 6, c * 30 + 5
            canvas[target_y:target_y+28, target_x:target_x+20] = cell

    text = pytesseract.image_to_string(canvas, config=config_block)
    digits = [int(ch) for ch in text if ch.isdigit()]
    
    # 숫자가 정확히 170개라면 바로 성공!
    if len(digits) == 170:
        print("Success!")
        board = []
        for r in range(ROWS):
            board.append(digits[r*COLS : (r+1)*COLS])
        return board, cropped_img
    
    # ==========================================
    # 3. [2단계] Retry Path: 행 단위로 다시 읽기 (더 정확함)
    # ==========================================
    print(f"Failed (Count: {len(digits)}). Switch to Row-by-Row scan.")
    board = []
    
    for r in range(ROWS):
        # 행 단위 이미지 생성 (H-Concat)
        row_imgs = cells_map[r] # 위에서 잘라둔 셀 활용
        
        # 간격(여백)을 두고 가로로 이어 붙이기
        row_strip = row_imgs[0]
        for c in range(1, COLS):
            # 구분선을 위한 흰색 여백 추가
            spacer = np.full((28, 10), 255, dtype=np.uint8) 
            row_strip = cv2.hconcat([row_strip, spacer, row_imgs[c]])
            
        text_row = pytesseract.image_to_string(row_strip, config=config_line)
        row_digits = [int(ch) for ch in text_row if ch.isdigit()]
        
        # ==========================================
        # 4. [3단계] Final Fallback: 칸 단위 읽기 (최후의 수단)
        # ==========================================
        # 행 단위 인식도 실패했다면(17개가 아니면), 그 행만 한 땀 한 땀 다시 읽음
        if len(row_digits) != 17:
            print(f"  -> Row {r} ambiguous. Switch to Cell-by-Cell.")
            row_digits = []
            for c in range(COLS):
                # 개별 셀 인식 (--psm 10)
                # 인식률 높이기 위해 테두리 여백을 좀 더 줌
                cell_padded = cv2.copyMakeBorder(cells_map[r][c], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255])
                char_text = pytesseract.image_to_string(cell_padded, config=config_char).strip()
                
                if char_text.isdigit():
                    row_digits.append(int(char_text))
                else:
                    # 정말로 인식이 안 되면 어쩔 수 없이 0 처리 (혹은 5로 추정)
                    # 하지만 psm 10은 거의 인식함
                    row_digits.append(0) 
        
        board.append(row_digits)
        
    return board, cropped_img
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
