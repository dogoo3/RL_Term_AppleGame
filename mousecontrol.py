import pyautogui
from pynput import mouse
import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import loadimagetogrid

# 전역 변수로 드래그 영역 좌표를 저장
drag_start_pos = None
drag_end_pos = None

left_click_pos = [0, 0]

def Click(x, y):
    """
    지정된 (x, y) 좌표 위치를 클릭.
    """
    print(f"클릭 수행: ({x}, {y})")
    pyautogui.click(x, y)

def SetArea(x1, y1, x2, y2):
    """
    드래그 시작과 끝 좌표를 전역 변수에 저장하고 출력.
    """
    global drag_start_pos, drag_end_pos
    drag_start_pos = (x1, y1)
    drag_end_pos = (x2, y2)
    
    print("--- 영역 좌표 저장 ---")
    print(f"시작 좌표 (좌측 상단): {drag_start_pos}")
    print(f"종료 좌표 (우측 하단): {drag_end_pos}")
    print("--------------------")

def Drag_pos(x1, y1, x2, y2, p_duration = 0.5):
    """
    (x1, y1)에서 (x2, y2)로 마우스를 드래그
    """
    print(f"\n드래그 : ({x1}, {y1}) -> ({x2}, {y2})")
    
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=p_duration, button='left')
    print("드래그 완료.")

def GetScreenShot(lu_x, lu_y, rd_x, rd_y):
    try:
        screenshot = pyautogui.screenshot(region=(lu_x,lu_y,rd_x-lu_x,rd_y-lu_y))
        screenshot.save("board.png")
        print("이미지 저장 완료: board.png")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")
        print("좌표가 화면 범위를 벗어났거나 너비/높이가 0 이하인지 확인하세요.")

def checkleftclick():
    with mouse.Listener(on_click=checkleftclick_position) as listener:
        listener.join()
        print(f"전역변수 : {left_click_pos}")
        return left_click_pos[:]
def checkleftclick_position(x, y, button, pressed):
    if pressed:
        if button == mouse.Button.left:
            print(f"좌클릭 위치 : {x}, {y}")
            left_click_pos[0] = x
            left_click_pos[1] = y
            return False

class SnippingTool: # 드래그하여 화면 영역을 선택하는 클래스.
    def __init__(self):
        self.root = tk.Tk()
        # 화면 전체 크기 설정 및 테두리 제거
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 1.0)  # 투명도 없음 (이미지로 처리)

        # 현재 화면 캡처 (밝은 원본)
        self.original_image = pyautogui.screenshot()
        
        # 어두운 배경 이미지 생성 (밝기 50% 감소)
        enhancer = ImageEnhance.Brightness(self.original_image)
        self.dark_image = enhancer.enhance(0.5)
        
        # Tkinter 호환 이미지로 변환
        self.tk_original = ImageTk.PhotoImage(self.original_image)
        self.tk_dark = ImageTk.PhotoImage(self.dark_image)

        # 캔버스 생성 (전체 화면)
        self.canvas = tk.Canvas(self.root, cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 처음에 어두운 이미지로 전체를 덮음
        self.canvas.create_image(0, 0, image=self.tk_dark, anchor="nw", tags="background")

        # 변수 초기화
        self.start_x = None
        self.start_y = None
        self.rect_id = None      # 빨간 테두리 ID
        self.bright_img_id = None # 밝은 영역 이미지 ID
        self.current_cropped_tk = None # 가비지 컬렉션 방지용 레퍼런스
        self.selected_coordinates = None

        # 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # ESC 키로 취소 기능 추가
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_button_press(self, event):
        # 시작 좌표 저장
        self.start_x = event.x
        self.start_y = event.y

    def on_move_press(self, event):
        cur_x, cur_y = event.x, event.y

        # 드래그 범위 계산 (좌상단, 우하단 정렬)
        x1, x2 = sorted([self.start_x, cur_x])
        y1, y2 = sorted([self.start_y, cur_y])

        # 기존 드래그 시각 요소 삭제
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.bright_img_id:
            self.canvas.delete(self.bright_img_id)

        # 1. 밝은 영역(선택 영역) 그리기
        # 원본 이미지에서 해당 부분만 잘라내어 캔버스에 덧붙임
        if x2 - x1 > 0 and y2 - y1 > 0:
            cropped = self.original_image.crop((x1, y1, x2, y2))
            self.current_cropped_tk = ImageTk.PhotoImage(cropped)
            self.bright_img_id = self.canvas.create_image(x1, y1, image=self.current_cropped_tk, anchor="nw")

        # 2. 빨간 테두리 그리기
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def on_button_release(self, event):
        # 최종 좌표 계산
        x1, x2 = sorted([self.start_x, event.x])
        y1, y2 = sorted([self.start_y, event.y])

        # 너비나 높이가 0인 경우 (그냥 클릭만 한 경우 등) 예외 처리
        if x2 - x1 < 5 or y2 - y1 < 5:
            print("영역이 너무 작습니다. 다시 시도해주세요.")
            return

        self.selected_coordinates = [(x1, y1), (x2, y2)]
        self.root.destroy() # 창 닫기

    def run(self):
        self.root.mainloop()
        return self.selected_coordinates

def Drag_User_Region(): # 사용자가 드래그하여 영역을 선택하도록 하고, 좌표를 반환
    print("화면을 드래그하여 영역을 선택하세요...")
    tool = SnippingTool()
    coords = tool.run()
    
    if coords:
        print("\n--- 영역 선택 완료 ---")
        print(f"시작 좌표: {coords[0]}")
        print(f"종료 좌표: {coords[1]}")
        print("--------------------")
        return coords
    else:
        print("영역 선택이 취소되었습니다.")
        return None

# --- 함수 사용 예제 ---
if __name__ == "__main__":
    # 기존 Click_User_Twice 대신 Drag_User_Region 사용
    positions = Drag_User_Region()
    
    # 반환된 좌표를 변수에 저장하여 활용
    if positions:
        pos1, pos2 = positions
        print(f"\n반환된 값 확인: pos1={pos1}, pos2={pos2}")

        left = pos1[0]
        top = pos1[1]
        width = pos2[0] - pos1[0]
        height = pos2[1] - pos1[1]
    
        try:
            # 선택한 영역 캡처 및 저장
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot.save("board.png")
            print("이미지 저장 완료: board.png")
        except Exception as e:
            print(f"이미지 저장 중 오류 발생: {e}")
            print("좌표가 화면 범위를 벗어났거나 너비/높이가 0 이하인지 확인하세요.")

        # Grid 인식 로직 실행
        try:
            grid = loadimagetogrid.recognize_digits_by_grid("board.png", {
            "1": "1.png", "2": "2.png", "3": "3.png",
            "4": "4.png", "5": "5.png", "6": "6.png",
            "7": "7.png", "8": "8.png", "9": "9.png"})

            if grid:
                print(grid[0][0])
                print("--- 인식된 숫자 그리드 ---")
                for row in grid:
                    print(row)
                print("------------------------")
        except Exception as e:
            print(f"그리드 인식 중 오류 발생 (loadimagetogrid 모듈 확인 필요): {e}")