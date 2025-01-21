import gym
import numpy as np
import random

class HuaRongDao(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(HuaRongDao, self).__init__()
        self.modules = {
            1: (2, 2),
            2: (2, 1),
            3: (1, 2),
            4: (1, 2),
            5: (1, 2),
            6: (1, 2), 
            7: (1, 1),
            8: (1, 1), 
            9: (1, 1), 
            10: (1, 1)
        }
        self.state = self.generate_initial_state()
        self.action_space = gym.spaces.Discrete(4 * len(self.modules))  # 4个方向 * N个模块
        self.observation_space = gym.spaces.Box(low=0, high=len(self.modules), shape=(20, ), dtype=np.int32)

    def generate_initial_state(self):
        state = np.zeros((5, 4), dtype=np.int32)
        state[0, 0] = 3
        state[0, 1] = 4
        state[0, 2] = 5
        state[0, 3] = 6
        state[1, 0] = 3
        state[1, 1] = 4
        state[1, 2] = 5
        state[1, 3] = 6
        state[2, 0] = 1
        state[2, 1] = 1
        state[2, 2] = 7
        state[2, 3] = 8
        state[3, 0] = 1
        state[3, 1] = 1
        state[3, 2] = 2
        state[3, 3] = 2
        state[4, 0] = 0
        state[4, 1] = 0
        state[4, 2] = 9
        state[4, 3] = 10
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
        # print("Reset:")
        # self.render()
        return self.state.flatten()

    def step(self, action):
        module_id = action // 4 + 1
        direction = ['up', 'down', 'left', 'right'][action % 4]
        
        if self.is_valid_move(module_id, direction):
            new_state = self.state.copy()
            self.move_module(module_id, direction)
            reward = 1.0 if np.all(self.state[3:5, 1:3] == 1) else 0.0
            done = reward == 1.0
        else:
            reward = -0.1
            done = False
            new_state = self.state.copy()
        
        return new_state.flatten(), reward, done, {}

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
        
         # 检查新位置是否超出棋盘边界
        for pos in new_positions:
            if pos[0] < 0 or pos[0] >= 5 or pos[1] < 0 or pos[1] >= 4:
                return False
        
        # 检查移动路径上是否有其他模块阻挡
        for pos in new_positions:
            if self.state[pos[0], pos[1]] != 0 and self.state[pos[0], pos[1]] != module_id:
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
        # print(f"render: close:{close}, mode:{mode}")
        if close:
            return
        # if mode == 'console':
        for row in self.state:
            print(" ".join(map(str, row)))
        print()

    def close(self):
        pass


from gym.envs.registration import registry

# 删除已注册的环境
env_id = 'HuaRongDao-v0'
if env_id in registry.env_specs:
    del registry.env_specs[env_id]

# 注册环境
gym.register(
    id='HuaRongDao-v0',
    entry_point='hrd:HuaRongDao',  # 确保模块名和类名正确
    max_episode_steps=1000,
)
