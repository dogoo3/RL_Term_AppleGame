# 강화학습개론 term project : Applegame

## 사과게임이란?
일본에서 만든 플래시게임이며, 가로 17개, 세로 10개로 배열된 게임판에서 화면상을 드래그하면 나타나는 사각형으로 감싼 사과 숫자의 합이 10이 되도록 둘러싸는 퍼즐 게임입니다. 

사각형으로 둘러싼 숫자의 합이 10이 되면 해당 사과들은 없어집니다.

사과의 위치가 떨어져 있어도 숫자의 합이 10이 되면 점수를 획득할 수 있습니다.

제한 시간은 120초이며, 게임 링크는 아래와 같습니다.

https://www.gamesaien.com/game/fruit_box_a/

## term project의 목적
- 강화학습 기법 중 하나를 적용하여, 게임의 주요 rule인 사각형으로 둘러싼 사과의 숫자 합이 10이 되는 조합을 찾습니다.
- 제한시간 내 혹은 주어진 게임판에서 가장 점수가 높게 나오는 로직으로 조합을 찾아 사과를 없앱니다.

## 실행 방법
Virtual environment를 생성한 뒤, 아래 명령어로 package를 설치해 줍니다.
```
pip install -r requirements.txt
```

아래 명령어로 프로그램을 실행합니다.
```
python main.py
```

처음 실행하는 경우, 1~4를 실행해 초기 좌표 설정을 진행합니다.<br>초기설정 영상은 아래를 참조하면 됩니다.

5번을 실행해 학습을 진행합니다.<br>학습이 완료된 뒤, reports 폴더에 학습 결과 그래프가 저장되고, models 폴더에 학습된 모델이 저장됩니다.

6번을 실행해 학습된 모델을 기반으로 게임을 진행합니다.

## 구동 영상

### 초기설정 가이드 및 첫 번째 실행 영상
[![Video Label](https://img.youtube.com/vi/uSM_3xvVKmQ/0.jpg)](https://youtu.be/uSM_3xvVKmQ)
### 두 번째 실행 영상
[![Video Label](https://img.youtube.com/vi/VWmWSdn7H_o/0.jpg)](https://youtu.be/VWmWSdn7H_o)

### 세 번째 실행 영상(Perfect)
[![Video Label](https://img.youtube.com/vi/L6qH2Jf8rcw/0.jpg)](https://youtu.be/L6qH2Jf8rcw)

## 주의점
Windows OS를 기반으로 제작하여 MacOS에서의 화면 캡쳐 및 마우스 조작이 가능한지는 검토할 수 없었습니다.