import os
import time
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import StepAPICompatibility

def make_env():
    def _thunk():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        class GymnasiumCompatibilityWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                kwargs.pop('seed', None)
                kwargs.pop('options', None)
                obs = self.env.reset(**kwargs)
                return obs, {}

            def step(self, action):
                obs, reward, done, info = self.env.step(action.item())
                return obs, reward, done, False, info

        env = GymnasiumCompatibilityWrapper(env)
        return env
    return _thunk

if __name__ == "__main__":
    # 병렬 환경 수
    NUM_ENVS = 8

    # SubprocVecEnv로 병렬 환경 구성
    envs = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # PPO 모델 정의
    model = PPO(
        policy="CnnPolicy",
        env=envs,
        verbose=1,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
    )

    # 학습 실행
    model.learn(total_timesteps=2_000_000)
    model.save("ppo_mario_multi")

    # 단일 환경으로 로드해 리플레이
    env = make_env()()
    model = PPO.load("ppo_mario_multi", device='mps' if torch.backends.mps.is_available() else 'cpu')

    obs, _ = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs.copy())
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.02)
        if done:
            obs, _ = env.reset()