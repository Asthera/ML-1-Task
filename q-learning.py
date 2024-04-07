import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self,
                 env: gym.wrappers,
                 gamma: float,
                 alpha: float,
                 epsilon: float,
                 decay_rate: float,
                 use_random_values: bool,
                 obs_space_steps: [int, int],
                 action_space_steps: int):

        self.env = env

        # define hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.use_random_values = use_random_values

        # define counts of states and actions (bins)
        self.obs_space_steps = obs_space_steps
        self.action_space_steps = action_space_steps

        # define spaces
        self.action_space = np.linspace(-1.0, 1.0, action_space_steps)
        self.position_space = np.linspace(-1.2, 0.6, obs_space_steps[0])
        self.velocity_space = np.linspace(-0.07, 0.07, obs_space_steps[1])

        # initialize Q(s, a)
        if self.use_random_values:
            self.Q = np.random.rand(obs_space_steps[0], obs_space_steps[1], action_space_steps)
        else:
            self.Q = np.zeros((obs_space_steps[0], obs_space_steps[1], action_space_steps))

        # steps, rewards to then plot it in the end
        self.steps = []
        self.rewards = []

    def train_episode(self):
        state, info = self.env.reset()
        state = self.discretize_state(state)

        done = False
        steps = 0
        rewards = 0

        while not done:
            # choose action using epsilon-greedy policy
            action_index = self.choose_action(state)

            action = self.action_space[action_index]

            # take action
            state_, reward, done, truncated, info = self.env.step([action])
            state_ = self.discretize_state(state_)

            # update Q(s, a)
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(a)(Q(s_, a)) - Q(s, a))
            self.Q[state[0], state[1], action_index] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[state_[0], state_[1], :]) - self.Q[
                state[0], state[1], action_index])

            state = state_

            steps += 1
            rewards += reward

        return steps, rewards

    def discretize_state(self, state: (float, float)):

        position = np.digitize(state[0], self.position_space)
        velocity = np.digitize(state[1], self.velocity_space)

        return position, velocity

    def choose_action(self, state: (int, int)):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.action_space_steps))
        else:
            return np.argmax(self.Q[state[0], state[1], :])

    def train(self, episodes: int):
        for episode in range(episodes):
            steps, rewards = self.train_episode()
            self.steps.append(steps)
            self.rewards.append(rewards)

            # if episode % 10 == 0:
            #     print("Episode: {} Steps: {}".format(episode, steps))

            self.decrease_epsilon()



    def decrease_epsilon(self):
        self.epsilon -= self.decay_rate
        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def save_weights(self, path: str):
        np.save(path, self.Q)

    def load_weights(self, path: str):
        self.Q = np.load(path)

    def get_steps(self):
        return self.steps

    def get_rewards(self):
        return self.rewards

    def plot(self, metric: str = "Rewards" or "Steps"):

        if metric == "Rewards":
            value = self.get_rewards()

        elif metric == "Steps":
            value = self.get_steps()

        else:
            raise ValueError("Invalid metric")

        plt.plot(value, label=metric)

        plt.title('Q-Learning')

        plt.xlabel('Episode')
        plt.ylabel('Value')

        plt.legend()
        plt.show()

    def get_average_reward(self):
        return np.mean(self.rewards)

    def get_average_steps(self):
        return np.mean(self.steps)

    def get_max_reward(self):
        return np.max(self.rewards)

    def get_success_rate(self):
        return np.sum(np.array(self.rewards) > -200) / len(self.rewards)


# Define hyperparameters
# [experiment, alpha, gamma, epsilon, decay rate, episodes, use_random_values, action_space_steps, obs_space_steps]
hyperparameters = [
    [1, 0.1, 0.8, 0.8, 0.001, 500, False, 5, [20, 20]],
    [2, 0.1, 0.9, 0.6, 0.0001, 500, True, 5, [50, 50]],
    [3, 0.1, 0.99, 0.5, 0.0001, 500, False, 5, [50, 50]],
    [4, 0.2, 0.8, 0.5, 0.0001, 500, True, 5, [100, 100]],
    [5, 0.2, 0.9, 0.65, 0.0001, 500, False, 5, [40, 40]],
    [6, 0.2, 0.99, 0.4, 0.0001, 500, True, 3, [100, 100]],
    [7, 0.3, 0.8, 0.5, 0.0001, 500, False, 10, [20, 20]],
    [8, 0.3, 0.9, 0.45, 0.0001, 500, True, 5, [50, 50]],
    [9, 0.3, 0.99, 0.4, 0.0001, 500, False, 9, [20, 20]],
    [10, 0.4, 0.8, 0.5, 0.0001, 1000, True, 10, [20, 20]],
    [11, 0.4, 0.9, 0.55, 0.0001, 1000, False, 5, [50, 50]],
    [12, 0.4, 0.99, 0.8, 0.0009, 1000, True, 3, [50, 50]],
    [13, 0.5, 0.8, 0.6, 0.0005, 1000, False, 10, [20, 20]],
    [14, 0.5, 0.9, 0.45, 0.0005, 1000, True, 5, [50, 50]],
    [15, 0.5, 0.99, 0.4, 0.0005, 1000, False, 9, [50, 50]]
]


def test_one_hyperparameter(hyperparameter: [int, float, float, float, float, int, bool, int, [int, int]]):
    env = gym.make("MountainCarContinuous-v0")

    # Retrieve hyperparameters from the list
    gamma = hyperparameter[2]
    alpha = hyperparameter[1]
    epsilon = hyperparameter[3]
    decay_rate = hyperparameter[4]
    use_random_values = hyperparameter[6]
    action_space_steps = hyperparameter[7]
    obs_space_steps = hyperparameter[8]

    # Train agent
    agent = QLearning(env, gamma, alpha, epsilon, decay_rate, use_random_values, obs_space_steps, action_space_steps)
    agent.train(hyperparameter[5])
    agent.save_weights(f"weights/qlearning/{hyperparameter[0]}.npy")
    env.close()

    # Calculate metrics
    average_reward = agent.get_average_reward()
    max_reward = agent.get_max_reward()
    average_steps = agent.get_average_steps()
    success_rate = agent.get_success_rate()

    return average_reward, max_reward, average_steps, success_rate


if __name__ == "__main__":

    print("Q-Learning with different hyperparameters")
    print("Hyperparameters: [experiment, alpha, gamma, epsilon, decay rate, episodes, use_random_values, action_space_steps, obs_space_steps]")

    for idx, hyperparameter in enumerate(hyperparameters):
        print(f"Hyperparameters {hyperparameter} ")
        metrics = test_one_hyperparameter(hyperparameter)
        print(f"Average reward: {metrics[0]:.2f}")
        print(f"Max reward: {metrics[1]:.2f}")
        print(f"Average steps: {metrics[2]:.2f}")
        print(f"Success rate: {metrics[3]:.2f}")
        print(f"---------------------{idx}/{len(hyperparameters)}------------------------")
