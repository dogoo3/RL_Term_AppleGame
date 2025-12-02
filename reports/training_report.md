# 사과게임 강화학습 보고서

## 사용한 강화학습 기법
- Deep Q-Network (DQN)을 사용하여 사각형 행동 공간을 학습했습니다.

## 학습 수식
DQN은 아래의 Bellman optimality를 목표로 하여 가중치를 업데이트합니다.

\[ y = r + \gamma \max_{a'} Q_{\text{target}}(s', a') \]

위 수식은 `reinforcement.py`의 `DQNAgent.train_step`에서 아래 코드로 구현되었습니다.

```python
with torch.no_grad():
    next_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
    targets = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
loss = self.criterion(q_values, targets)
```

## 학습 결과
- 총 Episode: 5000
- 평균 Reward: -1379.26
- 최고 Reward: -67.70
- 학습 시간: 19862.1 초

![training curve](training_curve.png)

## 추가 메모
- 합이 10인 사각형만을 선택하도록 환경에서 보상 함수를 설계했습니다.
- 학습이 완료된 모델은 `models/apple_dqn.pt` 경로에 저장되며 실게임 수행 시 로드됩니다.