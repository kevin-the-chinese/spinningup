import sys
import os
import gym

# 添加包含hrd.py的目录到Python模块路径中
sys.path.append('/workspaces/spinningup/spinup/usr_envs')
import hrd

# 尝试创建环境
try:
    env = gym.make('HuaRongDao-v0')
    print("环境创建成功！")
    # 重置环境并打印初始状态
    state = env.reset()
    print("初始状态:")
    env.render()
except gym.error.UnregisteredEnv as e:
    print(f"环境注册失败: {e}")
except Exception as e:
    print(f"发生错误: {e}")

# # 创建环境
# env = gym.make('HuaRongDao-v0')

# # 重置环境
# state = env.reset()
# print("重置后的状态:")
# env.render()

# while True:
#     # 捕获键盘输入
#     user_input = input("请输入模块ID和动作(格式:id action,输入q退出): ").strip()
    
#     if user_input.lower() == 'q':  # 如果输入q，则退出
#         print("退出程序。")
#         break
    
#     try:
#         # 解析输入
#         module_id, action = map(int, user_input.split())
        
#         # 执行动作
#         state, reward, done, info = env.step((module_id - 1) * 4 + action)
#         print(f"执行动作: 模块ID={module_id}, 动作={action}, 奖励: {reward}, 是否结束: {done}")
#         env.render()
        
#         # if done:
#         #     print("游戏结束。")
#         #     break
#     except ValueError:
#         print("输入格式错误，请输入两个整数（模块ID和动作），或者输入q退出。")
#     except Exception as e:
#         print(f"发生错误: {e}")
        
# # 关闭环境
# env.close()