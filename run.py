import retro

env = retro.make(game='SuperMarioBros-Nes')

obs = env.reset()

done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())  # random actions
    env.render()

env.close()
