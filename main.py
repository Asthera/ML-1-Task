import gymnasium as gym
from time import time
import numpy as np

def run():
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    action_value_left = 0.1
    action_value_right = 0.1

    # we need to rewrite it to "(1, )" shape
    action_value_left = np.array([action_value_left])
    action_value_right = np.array([action_value_right])
    print(action_value_left.shape)

    action_space = [action_value_left, action_value_right]

    state = env.reset()
    terminated = False

    start_time = time()

    while not terminated:

        # if more than 10 seconds have passed, make the car go right or left

        action = action_value_right
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)




    print(state)
    env.render()

if __name__ == '__main__':
    run()

