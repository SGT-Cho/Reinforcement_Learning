import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import time

NUM_ENVS = 8  # 실험적으로 줄여서 테스트하세요.
GAME_SPEED = 10.0
STEPS_PER_EPISODE = 5000

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    env.render(mode='human')  # 창 열기
    return env

envs = [make_env() for _ in range(NUM_ENVS)]

for step in range(STEPS_PER_EPISODE):
    for env in envs:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()
    time.sleep(0.02 / GAME_SPEED)

for env in envs:
    env.close()