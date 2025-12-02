import time
import loadimagetogrid
import mousecontrol
import manageposition
import reinforcement

GRID_RANGE_LU = [686, 269] # 범위 좌상단
GRID_RANGE_RD = [1819, 931] # 범위 우하단
START_BUTTON_POSITION = [983, 647] # 시작버튼
EXIT_BUTTON_POSITION = [1229, 747] # 종료버튼
RESET_BUTTON_POSITION = [1000, 500] # 리셋버튼

if __name__ == "__main__":
    position = manageposition.load_position()
    GRID_RANGE_LU = position[0]
    GRID_RANGE_RD = position[1]
    START_BUTTON_POSITION = position[2]
    EXIT_BUTTON_POSITION = position[3]
    RESET_BUTTON_POSITION = position[4]

    print(f"불러온 값 : {GRID_RANGE_LU},{GRID_RANGE_RD},{START_BUTTON_POSITION},{EXIT_BUTTON_POSITION},{RESET_BUTTON_POSITION}")

    print("---메뉴 선택---")
    print("1. 시작버튼 위치 조정")
    print("2. 종료버튼 위치 조정")
    print("3. 리셋버튼 위치 조정")
    print("4. 게임화면 범위 조정")
    print("5. 강화학습 시작")
    print("6, 학습된 모델로 실제 게임하기")
    print("7. 종료")

    while True:
        choice = input("번호를 입력하세요(1~7) : ")
        
        if choice == '1':
            print("시작 버튼을 클릭하세요!")
            click_pos = mousecontrol.checkleftclick()
            START_BUTTON_POSITION = click_pos
            manageposition.save_position(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, EXIT_BUTTON_POSITION, RESET_BUTTON_POSITION)
            print("시작 버튼이 저장되었습니다.")
        elif choice == '2':
            print("종료 버튼을 클릭하세요!")
            click_pos = mousecontrol.checkleftclick()
            EXIT_BUTTON_POSITION = click_pos
            manageposition.save_position(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, EXIT_BUTTON_POSITION, RESET_BUTTON_POSITION)
            print("종료 버튼이 저장되었습니다.")
        elif choice == '3':
            print("리셋 버튼을 클릭하세요!")
            click_pos = mousecontrol.checkleftclick()
            RESET_BUTTON_POSITION = click_pos
            manageposition.save_position(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, EXIT_BUTTON_POSITION, RESET_BUTTON_POSITION)
            print("리셋 버튼이 저장되었습니다.")
        elif choice == '4':
            area = mousecontrol.Drag_User_Region()
            GRID_RANGE_LU = area[0]
            GRID_RANGE_RD = area[1]
            manageposition.save_position(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, EXIT_BUTTON_POSITION, RESET_BUTTON_POSITION)

            mousecontrol.GetScreenShot(area[0][0], area[0][1], area[1][0], area[1][1])

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
                    print("게임화면 범위가 조정되었습니다! 인식된 숫자 그리드를 확인하세요!")
            except Exception as e:
                print(f"그리드 인식 중 오류 발생 (loadimagetogrid 모듈 확인 필요): {e}")


        elif choice == '5':
            while True:
                episode_input = input("학습 episode 수를 입력하세요(1 이상의 자연수): ")
                try:
                    episode_count = int(episode_input)
                    if episode_count < 1:
                        raise ValueError
                    break
                except ValueError:
                    print("episode 수는 1 이상의 정수로 입력하세요.")

            learning_obj = reinforcement.Reinforcement(episode_count)
            learning_obj.Init(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, RESET_BUTTON_POSITION)

            # episode 입력값 기반으로 강화학습을 수행한다
            learning_obj.load()
        elif choice == '6':
            learning_obj = reinforcement.Reinforcement(1)
            learning_obj.Init(GRID_RANGE_LU, GRID_RANGE_RD, START_BUTTON_POSITION, RESET_BUTTON_POSITION)
            learning_obj.load_real()
        elif choice == '7':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 1~7 사이의 값을 입력해 주세요.")
    
