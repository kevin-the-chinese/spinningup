import gym
import numpy as np
import random

class HuaRongDao(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(HuaRongDao, self).__init__()
        self.modules = {
            1: (1, 1),
            2: (1, 1),
            3: (1, 1),
            4: (1, 1),
            5: (1, 1),
            6: (1, 1), 
            7: (2, 1),
            8: (2, 1), 
            9: (1, 2), 
            10: (1, 2), 
            11: (2, 2)            
        }
        self.state = self.generate_initial_state()
        self.action_space = gym.spaces.Discrete(4 * len(self.modules))  # 4个方向 * N个模块
        self.observation_space = gym.spaces.Box(low=0, high=len(self.modules), shape=(20, ), dtype=np.int32)

    def generate_initial_state(self):
        state = np.zeros((5, 4), dtype=np.int32)
        state[0, 0] = 7
        state[0, 1] = 9
        state[0, 2] = 9
        state[0, 3] = 1
        state[1, 0] = 7
        state[1, 1] = 10
        state[1, 2] = 10
        state[1, 3] = 2
        state[2, 0] = 6
        state[2, 1] = 0
        state[2, 2] = 0
        state[2, 3] = 3
        state[3, 0] = 8
        state[3, 1] = 11
        state[3, 2] = 11
        state[3, 3] = 4
        state[4, 0] = 8
        state[4, 1] = 11
        state[4, 2] = 11
        state[4, 3] = 5
        return state

    def reset(self):
        self.state = self.generate_initial_state()
        M = random.randint(10, 100)  # 随机选择10到100之间的步数
        for _ in range(M):
            while True:
                module_id = random.choice(list(self.modules.keys()))
                direction = random.choice(['up', 'down', 'left', 'right'])
                if self.is_valid_move(module_id, direction):
                    self.move_module(module_id, direction)
                    break  # 退出内层循环，继续下一次移动
        return self.state.flatten()

    def step(self, action):
        module_id = action // 4 + 1
        direction = ['up', 'down', 'left', 'right'][action % 4]
        
        if self.is_valid_move(module_id, direction):
            new_state = self.state.copy()
            self.move_module(module_id, direction)
            reward = 1 if np.all(self.state[3:5, 1:3] == 11) else 0
            done = reward == 1
        else:
            reward = -0.1
            done = False
            new_state = self.state.copy()
        
        return new_state.flatten(), reward, done, {}, {}

    def is_valid_move(self, module_id, direction):
        module_size = self.modules[module_id]
        module_positions = np.argwhere(self.state == module_id)
        
        if direction == 'up':
            new_positions = module_positions - [1, 0]
        elif direction == 'down':
            new_positions = module_positions + [1, 0]
        elif direction == 'left':
            new_positions = module_positions - [0, 1]
        elif direction == 'right':
            new_positions = module_positions + [0, 1]
        
        for pos in new_positions:
            if pos[0] < 0 or pos[0] >= 5 or pos[1] < 0 or pos[1] >= 4 or self.state[pos[0], pos[1]] != 0:
                return False
        return True

    def move_module(self, module_id, direction):
        module_size = self.modules[module_id]
        module_positions = np.argwhere(self.state == module_id)
        
        if direction == 'up':
            new_positions = module_positions - [1, 0]
        elif direction == 'down':
            new_positions = module_positions + [1, 0]
        elif direction == 'left':
            new_positions = module_positions - [0, 1]
        elif direction == 'right':
            new_positions = module_positions + [0, 1]
        
        self.state[module_positions[:, 0], module_positions[:, 1]] = 0
        self.state[new_positions[:, 0], new_positions[:, 1]] = module_id

    def render(self, mode='console', close=False):
        if close:
            return
        if mode == 'console':
            for row in self.state:
                print(" ".join(map(str, row)))
            print()

    def close(self):
        pass


# 注册环境
gym.register(
    id='HuaRongDao-v0',
    entry_point='hrd:HuaRongDao',  # 确保模块名和类名正确
    max_episode_steps=1000,
)

# 创建环境
env = gym.make('HuaRongDao-v0')

# 重置环境
state = env.reset()
print("重置后的状态:")
env.render()

# 执行随机动作
for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, info, _ = env.step(action)
    module_id = action // 4 + 1
    direction = ['up', 'down', 'left', 'right'][action % 4]
    print(f"执行动作: {module_id, direction}, 奖励: {reward}, 是否结束: {done}")
    env.render()
    if done:
        break

# 关闭环境
env.close()