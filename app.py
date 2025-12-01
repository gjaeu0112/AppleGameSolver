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

def extract_grid_precise(img_stream):
    # 1. 이미지 로드 및 전처리
    file_bytes = np.frombuffer(img_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이진화 (배경/글씨=흰색, 사과=검은색)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # 배경 지우기 (Flood Fill) -> 사과 속 글씨만 남김
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    flooded = binary.copy()
    cv2.floodFill(flooded, mask, (0,0), 0)
    cv2.floodFill(flooded, mask, (w-1, h-1), 0)
    
    # 색상 반전 (흰 배경 검은 글씨)
    final_img = cv2.bitwise_not(flooded)
    
    # 글씨 영역 타이트하게 크롭
    temp_inv = cv2.bitwise_not(final_img)
    points = cv2.findNonZero(temp_inv)
    if points is not None:
        bx, by, bw, bh = cv2.boundingRect(points)
        final_img = final_img[by:by+bh, bx:bx+bw]
        # 원본 이미지도 나중에 결과 그릴 때 쓰려고 같이 자름 (좌표 매칭용)
        display_img = img[by:by+bh, bx:bx+bw]
    else:
        display_img = img

    # ---------------------------------------------------------
    # [핵심] 아틀라스(Atlas) 생성: 잘라서 새 판에 옮겨심기
    # ---------------------------------------------------------
    ROWS, COLS = 10, 17
    
    # 원본에서의 셀 크기
    cell_h = final_img.shape[0] / ROWS
    cell_w = final_img.shape[1] / COLS
    
    # 새로 만들 캔버스 설정 (한 글자당 28x28 크기로 규격화 + 여백)
    # 가로 간격을 넉넉히 줘서 숫자가 붙지 않게 함
    NEW_W, NEW_H = 28, 28
    GAP_X, GAP_Y = 15, 10
    
    canvas_width = COLS * (NEW_W + GAP_X)
    canvas_height = ROWS * (NEW_H + GAP_Y)
    
    # 깨끗한 흰색 도화지 생성
    atlas_canvas = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    
    print("1. 이미지를 170조각으로 자르고 재배치 중...")
    
    for r in range(ROWS):
        for c in range(COLS):
            # 1) 원본에서 해당 칸 좌표 계산 (소수점 정밀도 유지하다가 자를 때 int 변환)
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            
            # 2) 칸 오려내기
            cell = final_img[y1:y2, x1:x2]
            
            # 3) 사과 껍질(테두리) 제거를 위해 안쪽만 살짝 파냄 (Crop Center)
            ch, cw = cell.shape
            if ch > 0 and cw > 0:
                py, px = int(ch * 0.15), int(cw * 0.15) # 15%씩 파냄
                cell = cell[py:ch-py, px:cw-px]
            
            # 4) 규격화 (28x28 리사이즈)
            if cell.size > 0:
                cell = cv2.resize(cell, (NEW_W, NEW_H))
                # 이진화 한번 더 해서 선명하게 (흐릿한 잔상 제거)
                _, cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY)
            else:
                cell = np.full((NEW_H, NEW_W), 255, dtype=np.uint8) # 빈칸이면 흰색
            
            # 5) 새 도화지(Atlas)의 정확한 위치에 풀로 붙이기
            ty = r * (NEW_H + GAP_Y)
            tx = c * (NEW_W + GAP_X)
            atlas_canvas[ty:ty+NEW_H, tx:tx+NEW_W] = cell

    # ---------------------------------------------------------
    # OCR 실행 (딱 1번 호출)
    # ---------------------------------------------------------
    print("2. Tesseract OCR 1회 실행...")
    # --psm 6: 하나의 균일한 텍스트 블록으로 인식
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(atlas_canvas, config=config)
    
    # 숫자만 추출
    all_digits = [int(ch) for ch in text if ch.isdigit()]
    
    print(f"3. 인식된 숫자: {len(all_digits)}개 (목표: 170)")
    
    # ---------------------------------------------------------
    # 데이터 보정 및 결과 반환
    # ---------------------------------------------------------
    target_count = ROWS * COLS
    
    # 개수가 안 맞으면 0으로 채우거나 자름 (비상 대책)
    if len(all_digits) < target_count:
        all_digits += [0] * (target_count - len(all_digits))
    elif len(all_digits) > target_count:
        all_digits = all_digits[:target_count]
        
    board = []
    for r in range(ROWS):
        board.append(all_digits[r*COLS : (r+1)*COLS])
    
    # 디버깅용 이미지 (재배치된 아틀라스 이미지를 보여줌 - 인식 잘 됐는지 확인 가능)
    debug_img = cv2.cvtColor(atlas_canvas, cv2.COLOR_GRAY2BGR)
    
    return board, display_img, debug_img
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
