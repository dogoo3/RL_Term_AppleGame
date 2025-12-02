import cv2
import numpy as np

def recognize_digits_by_grid(large_image_path, template_paths): # 큰 이미지를 10x17 그리드로 나누고, 각 셀에서 가장 일치하는 숫자를 찾아 반환합니다.
    # 1. 이미지 불러오기 (흑백으로)
    large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
    if large_image is None:
        print(f"오류: 큰 이미지를 불러올 수 없습니다: {large_image_path}")
        return None

    # 템플릿 이미지들 불러오기
    templates = {}
    for digit_str, path in template_paths.items():
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"오류: 템플릿 이미지를 불러올 수 없습니다: {path}")
            continue
        templates[int(digit_str)] = template

    # 2. 그리드 크기 및 각 셀의 크기 계산
    grid_rows, grid_cols = 10, 17
    img_height, img_width = large_image.shape
    cell_height = img_height // grid_rows
    cell_width = img_width // grid_cols
    
    # 결과를 저장할 2D 리스트 초기화
    recognized_grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

    # 3. 각 셀을 순회하며 숫자 인식
    for r in range(grid_rows):
        for c in range(grid_cols):
            # 현재 셀의 이미지 영역(ROI) 추출
            y_start, y_end = r * cell_height, (r + 1) * cell_height
            x_start, x_end = c * cell_width, (c + 1) * cell_width
            cell_roi = large_image[y_start:y_end, x_start:x_end]

            best_match_score = -1
            best_match_digit = -1

            # 4. 현재 셀(ROI)에서 모든 템플릿과 매칭 시도
            for digit, template in templates.items():
                # ROI가 템플릿보다 작은 경우를 방지
                if cell_roi.shape[0] < template.shape[0] or cell_roi.shape[1] < template.shape[1]:
                    continue

                res = cv2.matchTemplate(cell_roi, template, cv2.TM_CCOEFF_NORMED)
                _minVal, maxVal, _minLoc, _maxLoc = cv2.minMaxLoc(res)
                
                # 가장 높은 유사도를 가진 숫자를 찾음
                if maxVal > best_match_score:
                    best_match_score = maxVal
                    best_match_digit = digit
            
            # 해당 셀의 숫자를 그리드에 저장
            recognized_grid[r][c] = best_match_digit

    return recognized_grid

if __name__ == "__main__":
    large_image_file = "board_3.png"
    template_files = {
        "1": "1.png", "2": "2.png", "3": "3.png",
        "4": "4.png", "5": "5.png", "6": "6.png",
        "7": "7.png", "8": "8.png", "9": "9.png"
    }

    grid = recognize_digits_by_grid(large_image_file, template_files)

    if grid:
        print("--- 인식된 숫자 그리드 ---")
        for row in grid:
            # 이제 'row'는 항상 리스트이므로 오류가 발생하지 않습니다.
            print(" ".join(map(str, row)))
        print("------------------------")