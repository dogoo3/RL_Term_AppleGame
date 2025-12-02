import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import loadimagetogrid
import mousecontrol


class AppleGameEnv:
    """Lightweight simulator for the 17x10 apple board."""

    def __init__(self, rows: int = 10, cols: int = 17, max_steps: int = 120, seed: int = 42) -> None:
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps # 에피소드 1개당 최대 step
        self.random_state = np.random.default_rng(seed)
        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.step_count = 0
        self.action_rectangles: List[Tuple[int, int, int, int]] = self._build_action_space()

    def _build_action_space(self) -> List[Tuple[int, int, int, int]]:
        # 게임판에 있는 모든 축정렬 사각형을 나열함, 학습 공간을 정의
        rects: List[Tuple[int, int, int, int]] = []
        for top in range(self.rows):
            for left in range(self.cols):
                for bottom in range(top, self.rows):
                    for right in range(left, self.cols):
                        rects.append((top, left, bottom, right))
        return rects

    @property
    def action_size(self) -> int:
        return len(self.action_rectangles)

    @property
    def state_size(self) -> int:
        return self.rows * self.cols

    def reset(self, grid: Optional[List[List[int]]] = None) -> np.ndarray:
        if grid is None:
            self.grid = self.random_state.integers(1, 10, size=(self.rows, self.cols), dtype=np.int32)
        else:
            arr = np.array(grid, dtype=np.int32)
            if arr.shape != (self.rows, self.cols):
                raise ValueError("잘못된 grid 형태입니다.")
            self.grid = arr.copy()
        self.step_count = 0
        return self.grid_to_state(self.grid)

    def grid_to_state(self, grid: Union[List[List[int]], np.ndarray]) -> np.ndarray:
        arr = np.array(grid, dtype=np.float32)
        arr[arr < 0] = 0  # template matching 실패시 -1 보정
        return (arr / 10.0).flatten()

    def _sample_new_values(self, rect: Tuple[int, int, int, int]) -> np.ndarray:
        h = rect[2] - rect[0] + 1
        w = rect[3] - rect[1] + 1
        return self.random_state.integers(1, 10, size=(h, w), dtype=np.int32)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        rect = self.action_rectangles[action_index]
        top, left, bottom, right = rect
        subgrid = self.grid[top : bottom + 1, left : right + 1]
        non_zero_cells = np.count_nonzero(subgrid) # 카운트 개수
        rect_sum = int(np.sum(subgrid))

        reward = -0.2
        # print(f"rect_sum : {rect_sum}")
        if rect_sum == 0: # 빈 그리드를 드래그했을 때 감점을 매우 크게 준다
            reward = -20.0
            # print("빈 그리드를 드래그했다")
        elif rect_sum == 10:
            reward = 5.0 + float(non_zero_cells)
            self.grid[top : bottom + 1, left : right + 1] = 0
            # print(f"--- {self.step_count}회 게임판 상태 ---")
            # for row in self.grid:
            #     print(row)
            # print("맞추었다!")
            # time.sleep(2)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.grid_to_state(self.grid), reward, done, {"rect_sum": rect_sum, "rect": rect}

    def valid_actions_for_grid(self, grid: List[List[int]]) -> List[int]:
        arr = np.array(grid, dtype=np.int32)
        valid: List[int] = []
        for idx, (top, left, bottom, right) in enumerate(self.action_rectangles):
            subgrid = arr[top : bottom + 1, left : right + 1]
            if np.count_nonzero(subgrid) == 0: 
                continue
            if int(np.sum(subgrid)) == 10:
                valid.append(idx)
        return valid

    def rect_to_indices(self, rect: Tuple[int, int, int, int]) -> Tuple[List[int], List[int]]:
        top, left, bottom, right = rect
        return [left, top], [right, bottom]


class PrioritizedReplayBuffer:
    """Prioritized replay buffer with proportional sampling."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer: List[Optional[Tuple[np.ndarray, int, float, np.ndarray, bool]]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.frame = 1

    def __len__(self) -> int:
        return self.size

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        max_pr = self.priorities[: self.size].max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_pr if max_pr > 0 else 1.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.size]

        scaled_pr = priorities ** self.alpha
        total = scaled_pr.sum()
        if total <= 0:
            scaled_pr = np.ones_like(scaled_pr) / len(scaled_pr)
        else:
            scaled_pr = scaled_pr / total

        indices = np.random.choice(self.size, batch_size, p=scaled_pr)
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*samples)  # type: ignore[arg-type]

        beta = self._beta_by_frame()
        weights = (self.size * scaled_pr[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            np.stack(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.stack(next_state),
            np.array(done, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def _beta_by_frame(self) -> float:
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        return beta

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        errors = np.abs(errors) + 1e-5
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(err)


class QNetwork(nn.Module):
    """Fully connected value network."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    DQN agent adapted from lectures/ex011_cartpole_dqn.py with a larger output head
    that matches the AppleGame rectangle action space.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 60_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.15,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 10,
        prio_alpha: float = 0.6,
        prio_beta_start: float = 0.4,
        prio_beta_frames: int = 50_000,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, alpha=prio_alpha, beta_start=prio_beta_start, beta_frames=prio_beta_frames
        )
        self.train_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def select_best_from_valid(self, state: np.ndarray, valid_indices: List[int]) -> Optional[int]:
        if not valid_indices:
            return None
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        masked = np.full_like(q_values, -np.inf)
        masked[valid_indices] = q_values[valid_indices]
        best_index = int(np.argmax(masked))
        if not np.isfinite(masked[best_index]):
            return None
        return best_index

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.buffer.sample(self.batch_size)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            # Double DQN: select greedy actions with the policy net, evaluate with the target net.
            next_policy_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states_tensor).gather(1, next_policy_actions)
            targets = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        td_errors = targets - q_values
        loss = (weights_tensor * (td_errors ** 2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: Path) -> bool:
        if not path.exists():
            return False
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        return True


class Reinforcement:
    """High-level controller that trains and runs the DQN agent on the real game board."""

    def __init__(self, episode_count: int = 0) -> None:
        self.LU_X = 0
        self.LU_Y = 0
        self.RD_X = 0
        self.RD_Y = 0
        self.start_button_position = [0, 0]
        self.reset_button_position = [0, 0]
        self.grid_unit_size = 0
        self.grid: Optional[List[List[int]]] = None
        self.episode_count = max(0, int(episode_count))
        self.env = AppleGameEnv()
        self.agent = DQNAgent(self.env.state_size, self.env.action_size)
        self.model_path = Path("models/apple_dqn.pt")
        self.stats_path = Path("reports/training_stats.json")
        self.plot_path = Path("reports/training_curve.png")
        self.report_path = Path("reports/training_report.md")
        self.runtime_limit = 120
        self.max_live_actions = 150
        self.drag_delay = 0.3

    def Init(self, p_lu: List[int], p_rd: List[int], p_btn_start: List[int], p_btn_reset: List[int]) -> None:
        self.LU_X, self.LU_Y = p_lu
        self.RD_X, self.RD_Y = p_rd
        self.grid_unit_size = int((self.RD_X - self.LU_X) / 17)
        self.start_button_position[0] = p_btn_start[0]
        self.start_button_position[1] = p_btn_start[1]
        self.reset_button_position[0] = p_btn_reset[0]
        self.reset_button_position[1] = p_btn_reset[1]

    def _template_paths(self) -> Dict[str, str]:
        return {str(i): f"{i}.png" for i in range(1, 10)}

    def load(self) -> None:
        # self.grid = p_grid
        training_stats: Optional[Dict[str, List[float]]] = None
        if self.episode_count > 0:
            training_stats = self._train_agent(self.episode_count)
            if training_stats:
                self._persist_training_artifacts(training_stats)
        elif not self.model_path.exists():
            print("저장된 모델이 없습니다. 최소 1회 이상 episode를 학습하세요.")
            return

    def load_real(self):
        self.agent.load_model(self.model_path)
        self._play_live_game()

    def _train_agent(self, episode_count: int) -> Optional[Dict[str, List[float]]]: # 훈련 진행
        print(f"강화학습 진행: {episode_count} episodes")
        rewards: List[float] = []
        losses: List[float] = []
        epsilons: List[float] = []
        start_time = time.time()

        for episode in range(1, episode_count + 1):
            state = self.env.reset()
            episode_reward = 0.0
            episode_losses: List[float] = []

            done = False
            count = 0
            while not done:
                action = self.agent.select_action(state, explore=True)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.push(state, action, reward, next_state, done)
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                episode_reward += reward
                state = next_state

            self.agent.decay_epsilon()

            rewards.append(episode_reward)
            epsilons.append(self.agent.epsilon)
            if episode_losses:
                losses.append(float(np.mean(episode_losses)))
            else:
                losses.append(0.0)

            if episode % 10 == 0 or episode == episode_count:
                print(
                    f"Episode {episode:04d} | Reward {episode_reward:.2f} | "
                    f"Epsilon {self.agent.epsilon:.3f} | AvgLoss {losses[-1]:.4f}"
                )
            
            # episode가 80번 돌아갈 때마다 게임판을 reset하여 새 grid를 생성
            # if episode % 10 == 0 and episode < episode_count:
            #     mousecontrol.Click(self.reset_button_position[0], self.reset_button_position[1]) # 리셋 버튼 클릭
            #     time.sleep(0.5) # 화면 전환 딜레이 대기
            #     mousecontrol.Click(self.start_button_position[0], self.start_button_position[1]) # 시작 버튼 클릭
            #     time.sleep(0.1) # 화면 전환 딜레이 대기
            #     mousecontrol.GetScreenShot(self.LU_X, self.LU_Y, self.RD_X, self.RD_Y) # 게임판 화면 캡쳐
            #     self.grid = loadimagetogrid.recognize_digits_by_grid("board.png", { # 캡쳐화면을 가지고 새 게임판 제작
            #     "1": "1.png", "2": "2.png", "3": "3.png",
            #     "4": "4.png", "5": "5.png", "6": "6.png",
            #     "7": "7.png", "8": "8.png", "9": "9.png"})
            #     print("새 게임판 생성 완료")
            #     if self.grid:
            #         print("--- 새로 만들어진 숫자 그리드 ---")
            #         for row in self.grid:
            #             print(row)

        elapsed = time.time() - start_time
        self.agent.save_model(self.model_path)
        moving_avg = self._moving_average(rewards, window=10)
        stats = {
            "episode_rewards": rewards,
            "average_loss": losses,
            "epsilons": epsilons,
            "moving_average": moving_avg,
            "episode_count": episode_count,
            "training_time_sec": elapsed,
        }
        return stats

    def _moving_average(self, values: List[float], window: int) -> List[float]: 
        if not values:
            return []
        averages = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start : i + 1]
            averages.append(float(np.mean(window_values)))
        return averages

    def _persist_training_artifacts(self, stats: Dict[str, List[float]]) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self._plot_training_curve(stats)
        self._write_training_report(stats)

    def _plot_training_curve(self, stats: Dict[str, List[float]]) -> None:
        episodes = range(1, stats["episode_count"] + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, stats["episode_rewards"], label="Episode reward", alpha=0.4)
        plt.plot(episodes, stats["moving_average"], label="Moving average (10)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("AppleGame DQN Training")
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path)
        plt.close()

    def _write_training_report(self, stats: Dict[str, List[float]]) -> None:
        best_reward = max(stats["episode_rewards"]) if stats["episode_rewards"] else 0.0
        avg_reward = float(np.mean(stats["episode_rewards"])) if stats["episode_rewards"] else 0.0
        report_lines = [
            "# 사과게임 강화학습 보고서",
            "",
            "## 사용한 강화학습 기법",
            "- Deep Q-Network (DQN)을 사용하여 사각형 행동 공간을 학습했습니다.",
            "",
            "## 학습 수식",
            "Double DQN을 사용하여 정책망으로 행동을 고르고 타깃망으로 가치를 평가합니다.",
            "",
            r"\[ y = r + \gamma Q_{\text{target}}\!\bigl(s', \arg\max_{a'} Q_{\text{policy}}(s', a')\bigr) \]",
            "",
            "위 수식은 `reinforcement.py`의 `DQNAgent.train_step`에서 아래 코드로 구현되었습니다.",
            "",
            "```python",
            "with torch.no_grad():",
            "    next_policy_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)",
            "    next_q_values = self.target_net(next_states_tensor).gather(1, next_policy_actions)",
            "    targets = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)",
            "```",
            "",
            "Prioritized Experience Replay를 도입하여 TD-error가 큰 전이 샘플일수록 자주 학습하고,",
            "중복 샘플링 편향은 중요도 가중치로 보정했습니다.",
            "",
            "## 학습 결과",
            f"- 총 Episode: {stats['episode_count']}",
            f"- 평균 Reward: {avg_reward:.2f}",
            f"- 최고 Reward: {best_reward:.2f}",
            f"- 학습 시간: {stats.get('training_time_sec', 0):.1f} 초",
            "",
            "![training curve](training_curve.png)",
            "",
            "## 추가 메모",
            "- 합이 10인 사각형만을 선택하도록 환경에서 보상 함수를 설계했습니다.",
            "- 학습이 완료된 모델은 `models/apple_dqn.pt` 경로에 저장되며 실게임 수행 시 로드됩니다.",
        ]
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def _play_live_game(self) -> None: # 실제 게임 진행
        # 실시간 게임을 진행하기 이전 게임판을 reset한다.
        mousecontrol.Click(self.reset_button_position[0], self.reset_button_position[1]) # 리셋 버튼 클릭
        time.sleep(0.5) # 화면 전환 딜레이 대기
        mousecontrol.Click(self.start_button_position[0], self.start_button_position[1]) # 시작 버튼 클릭
        time.sleep(0.1) # 화면 전환 딜레이 대기
        mousecontrol.GetScreenShot(self.LU_X, self.LU_Y, self.RD_X, self.RD_Y) # 게임판 화면 캡쳐
        self.grid = loadimagetogrid.recognize_digits_by_grid("board.png", { # 캡쳐화면을 가지고 새 게임판 제작
        "1": "1.png", "2": "2.png", "3": "3.png",
        "4": "4.png", "5": "5.png", "6": "6.png",
        "7": "7.png", "8": "8.png", "9": "9.png"})
        print("새 게임판 생성 완료")
        if self.grid:
            print("--- 새로 만들어진 숫자 그리드 ---")
            for row in self.grid:
                print(row)

        if self.grid is None:
            print("실시간 게임을 위한 grid 정보가 없습니다.")
            return
        # grid = self._sanitize_grid(self.grid)
        print("학습된 정책으로 실시간 게임을 실행합니다.")
        start_time = time.time()
        actions_taken = 0
        recent_rects: Deque[Tuple[int, int, int, int]] = deque(maxlen=5)
        stale_rect_states: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}
        empty_rects: Set[Tuple[int, int, int, int]] = set()
        cleared_cells: Set[Tuple[int, int]] = set()

        while (time.time() - start_time) < self.runtime_limit and actions_taken < self.max_live_actions:
            valid_actions = self.env.valid_actions_for_grid(self.grid)
            if not valid_actions:
                print("합이 10인 조합이 더 이상 없습니다.")
                break

            action_index = self._select_live_action(
                self.grid, valid_actions, recent_rects, stale_rect_states, empty_rects, cleared_cells
            )
            if action_index is None:
                print("정책이 유효한 행동을 찾지 못했습니다.")
                break

            rect = self.env.action_rectangles[action_index]
            start_idx, end_idx = self.env.rect_to_indices(rect)
            rect_snapshot = self._rect_snapshot(self.grid, rect)
            if self._is_empty_snapshot(rect_snapshot):
                empty_rects.add(rect)
                continue
            # if not self._is_sum_ten(grid, rect):
            #     # grid 오차로 인해 잘못된 행동을 선택한 경우 최신 스크린샷을 확보한다.
            #     grid = self._capture_grid() or grid
            #     continue

            stale_rect_states[rect] = rect_snapshot
            if self._is_sum_ten(self.grid, rect):
                for cell in self._rect_cells(rect):
                    cleared_cells.add(cell)
            self.drag(start_idx, end_idx)
            self.check_ten(start_idx, end_idx)
            actions_taken += 1
            recent_rects.append(rect)
            time.sleep(self.drag_delay)
            # latest_grid = self._capture_grid()
            # if latest_grid is None:
            #     print("그리드 인식 실패로 자동 수행을 종료합니다.")
            #     break
            # grid = latest_grid
            updated_snapshot = self._rect_snapshot(self.grid, rect)
            if stale_rect_states.get(rect) != updated_snapshot:
                stale_rect_states.pop(rect, None)
            if self._is_empty_snapshot(updated_snapshot):
                empty_rects.add(rect)
            else:
                empty_rects.discard(rect)
            cleared_cells = self._filter_cleared_cells(cleared_cells, self.grid)

            
            if self.grid:
                print(f"--- {actions_taken}회차 그리드 ---")
                for row in self.grid:
                    print(row)

        print(f"자동 실행 종료 - 수행한 드래그 횟수: {actions_taken}")

    def _is_sum_ten(self, grid: List[List[int]], rect: Tuple[int, int, int, int]) -> bool:
        top, left, bottom, right = rect
        arr = np.array(grid, dtype=np.int32)
        subgrid = arr[top : bottom + 1, left : right + 1]
        return np.count_nonzero(subgrid) > 0 and int(np.sum(subgrid)) == 10

    def _capture_grid(self) -> Optional[List[List[int]]]:
        try:
            mousecontrol.GetScreenShot(self.LU_X, self.LU_Y, self.RD_X, self.RD_Y)
            grid = loadimagetogrid.recognize_digits_by_grid("board.png", self._template_paths())
            if grid:
                return self._sanitize_grid(grid)
        except Exception as exc:
            print(f"그리드 캡처 중 오류 발생: {exc}")
        return None

    def _sanitize_grid(self, grid: List[List[int]]) -> List[List[int]]:
        sanitized = []
        for row in grid:
            sanitized.append([int(val) if int(val) > 0 else 0 for val in row])
        return sanitized

    def _select_live_action( # 다음 액션을 선택하는 로직이 들어간 함수
        self,
        grid: List[List[int]],
        valid_actions: List[int],
        recent_rects: Deque[Tuple[int, int, int, int]],
        stale_rect_states: Dict[Tuple[int, int, int, int], Tuple[int, ...]],
        empty_rects: Set[Tuple[int, int, int, int]],
        cleared_cells: Set[Tuple[int, int]],
    ) -> Optional[int]:
        avoid_set = set(recent_rects)
        candidate_actions: List[int] = []
        for idx in valid_actions:
            rect = self.env.action_rectangles[idx]
            # print(rect)
            if rect in avoid_set:
                continue
            if rect in empty_rects:
                continue
            start_cell = (rect[0], rect[1])
            if start_cell in cleared_cells:
                continue
            if self._rect_contains_cleared(rect, cleared_cells):
                continue
            if self._rect_has_zero(grid, rect):
                continue
            snapshot = self._rect_snapshot(grid, rect)
            previous = stale_rect_states.get(rect)
            if previous is not None and previous == snapshot:
                continue
            candidate_actions.append(idx)
        if not candidate_actions:
            candidate_actions = valid_actions
        state = self.env.grid_to_state(grid)
        return self.agent.select_best_from_valid(state, candidate_actions)

    def _rect_snapshot(self, grid: List[List[int]], rect: Tuple[int, int, int, int]) -> Tuple[int, ...]:
        top, left, bottom, right = rect
        values: List[int] = []
        for row in range(top, bottom + 1):
            values.extend(grid[row][left : right + 1])
        return tuple(values)

    def _is_empty_snapshot(self, snapshot: Tuple[int, ...]) -> bool:
        return all(value == 0 for value in snapshot)

    def _rect_cells(self, rect: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        top, left, bottom, right = rect
        cells: List[Tuple[int, int]] = []
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                cells.append((row, col))
        return cells

    def _filter_cleared_cells(self, cleared: Set[Tuple[int, int]], grid: List[List[int]]) -> Set[Tuple[int, int]]:
        remaining: Set[Tuple[int, int]] = set()
        for cell in cleared:
            row, col = cell
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 0:
                remaining.add(cell)
        return remaining

    def _rect_contains_cleared(self, rect: Tuple[int, int, int, int], cleared: Set[Tuple[int, int]]) -> bool:
        top, left, bottom, right = rect
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if (row, col) in cleared:
                    return True
        return False

    def _rect_has_zero(self, grid:List[List[int]], rect:Tuple[int, int, int, int]) -> bool:
        top, left, bottom, right = rect
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                print(f"{row}-{col} : {grid[row][col]}")
                if grid[row][col] == 0:
                    return True
        return False

    def drag(self, start_grid_index: List[int], end_grid_index: List[int]) -> None:
        duration = (end_grid_index[0] - start_grid_index[0] + 1) * (end_grid_index[1] - start_grid_index[1] + 1)
        if duration <= 2:
            duration = 0.8
        elif duration <= 5:
            duration = 1.7
        else:
            duration = 2.3
        mousecontrol.Drag_pos(
            self.LU_X + self.grid_unit_size * start_grid_index[0],
            self.LU_Y + self.grid_unit_size * start_grid_index[1],
            self.LU_X + self.grid_unit_size * end_grid_index[0] + self.grid_unit_size,
            self.LU_Y + self.grid_unit_size * end_grid_index[1] + self.grid_unit_size,
            duration,
        )

    def check_ten(self, start_grid_index: List[int], end_grid_index: List[int], grid: Optional[List[List[int]]] = None):
        target_grid = grid or self.grid
        if target_grid is None:
            return False
        total = 0
        for i in range(start_grid_index[0], end_grid_index[0] + 1):
            for j in range(start_grid_index[1], end_grid_index[1] + 1):
                total += target_grid[j][i]
        
        if total == 10:
            for i in range(start_grid_index[0], end_grid_index[0] + 1):
                for j in range(start_grid_index[1], end_grid_index[1] + 1):
                    target_grid[j][i] = 0


    # def check_ten(
    #     self, start_grid_index: List[int], end_grid_index: List[int], grid: Optional[List[List[int]]] = None
    # ) -> bool:
    #     target_grid = grid or self.grid
    #     if target_grid is None:
    #         return False
    #     total = 0
    #     for i in range(start_grid_index[0], end_grid_index[0] + 1):
    #         for j in range(start_grid_index[1], end_grid_index[1] + 1):
    #             total += target_grid[j][i]
    #     return total == 10
