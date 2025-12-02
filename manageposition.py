def load_position():
    # 1. 파일 읽기 및 데이터 파싱
    coordinates = []
    try:
        with open("position.txt", 'r') as f:
            for line in f:
                # 줄바꿈 제거 후 쉼표(,)로 나누고 정수로 변환
                parts = line.strip().split(',')
                if len(parts) == 2:
                    x, y = int(parts[0]), int(parts[1])
                    coordinates.append([x, y])
    except FileNotFoundError:
        print(f"{input_filename} 파일이 없습니다. 파일을 먼저 생성해주세요.")
        return

    return coordinates

def save_position(lu, rd, start, exit, reset):
    with open("position.txt", 'w') as f:
        f.write(f"{lu[0]},{lu[1]}\n")
        f.write(f"{rd[0]},{rd[1]}\n")
        f.write(f"{start[0]},{start[1]}\n")
        f.write(f"{exit[0]},{exit[1]}\n")
        f.write(f"{reset[0]},{reset[1]}")
    print("저장 완료!")