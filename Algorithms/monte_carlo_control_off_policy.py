import numpy as np
import gymnasium as gym


class MonteCarlosControlOffPolicy:
    def __init__(self,
                 env: gym.wrappers,
                 gamma: float,
                 epsilon: float,
                 decay_rate: float,
                 use_random_values: bool,
                 obs_space_steps: [int, int],
                 action_space_steps: int):

        self.env = env

        # define hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.obs_space_steps = obs_space_steps
        self.action_space_steps = action_space_steps

        # define spaces
        self.action_space = np.linspace(-1.0, 1.0, action_space_steps)
        self.position_space = np.linspace(-1.2, 0.6, obs_space_steps[0])
        self.velocity_space = np.linspace(-0.07, 0.07, obs_space_steps[1])

        # initialize Q(s, a)
        if use_random_values:
            self.Q = np.random.rand(obs_space_steps[0], obs_space_steps[1], action_space_steps)
        else:
            self.Q = np.zeros((obs_space_steps[0], obs_space_steps[1], action_space_steps))

        # initialize C(s, a)
        self.C = np.zeros((obs_space_steps[0], obs_space_steps[1], action_space_steps))

        # initialize target policy
        self.pi = np.argmax(self.Q, axis=2)

        # steps, rewards to then plot it in the end
        self.steps = []
        self.rewards = []


    def generate_episode(self, policy_b):

        episode = []
        state, info = self.env.reset()
        state = self.discretize_state(state)

        done = False

        while not done:
            probs = policy_b[state[0], state[1], :]
            # print("Probs: ", probs)

            action_index = np.random.choice(np.arange(self.action_space_steps), p=probs)

            action = self.action_space[action_index]

            state_, reward, done, truncated, info = self.env.step([action])
            state_ = self.discretize_state(state_)
            episode.append((state, action_index, reward))
            state = state_

        self.rewards.append(sum([x[2] for x in episode]))
        self.steps.append(len(episode))

        return episode

    def train_on_episode(self, episode, policy_b):
        G = 0
        W = 1

        # Here we iterate over the episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            self.C[state[0], state[1], action] += W

            # update Q(s, a)
            # Q(s, a) = Q(s, a) + (W / C(s, a)) * (G - Q(s, a))
            self.Q[state[0], state[1], action] += (W / self.C[state[0], state[1], action]) * (
                    G - self.Q[state[0], state[1], action])

            self.pi[state[0], state[1]] = np.argmax(self.Q[state[0], state[1], :])

            if action != self.pi[state[0], state[1]]:
                # print(f"Action {action} != {self.pi[state[0], state[1]]}, breaking at {t}/{len(episode)}")
                break

            W = W / policy_b[state[0], state[1], action]

    def train(self, episodes, policy_b):
        for idx in range(episodes):
            episode = self.generate_episode(policy_b)
            self.train_on_episode(episode, policy_b)

            # print(f"Episode {idx}/{episodes} done")

    def discretize_state(self, state):
        position = np.digitize(state[0], self.position_space)
        velocity = np.digitize(state[1], self.velocity_space)
        return position, velocity

    def save_weights(self, path: str):
        np.save(path + "_Q" + ".npy", self.Q)
        np.save(path + "_policy" + ".npy", self.pi)
        np.save(path + "_C" + ".npy", self.C)


    def load_weights(self, path: str):
        self.Q = np.load(path + "_Q" + ".npy")
        self.pi = np.load(path + "_policy" + ".npy")
        self.C = np.load(path + "_C" + ".npy")

    def get_steps(self):
        return self.steps

    def get_rewards(self):
        return self.rewards

    def get_average_reward(self):
        return np.mean(self.rewards)

    def get_max_reward(self):
        return np.max(self.rewards)

    def get_average_steps(self):
        return np.mean(self.steps)

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
    epsilon = hyperparameter[3]
    decay_rate = hyperparameter[4]
    use_random_values = hyperparameter[6]
    action_space_steps = hyperparameter[7]
    obs_space_steps = hyperparameter[8]

    policy_b = np.full((obs_space_steps[0], obs_space_steps[1], action_space_steps), 1 / action_space_steps)

    # Train agent
    agent = MonteCarlosControlOffPolicy(env, gamma, epsilon, decay_rate, use_random_values, obs_space_steps,
                                        action_space_steps)
    agent.train(hyperparameter[5], policy_b)
    agent.save_weights(f"weights/mc_off_policy/{hyperparameter[0]}")
    env.close()

    # Calculate metrics
    average_reward = agent.get_average_reward()
    max_reward = agent.get_max_reward()
    average_steps = agent.get_average_steps()
    success_rate = agent.get_success_rate()

    return average_reward, max_reward, average_steps, success_rate


if __name__ == "__main__":

    print("Sarsa with different hyperparameters")
    print(
        "Hyperparameters: [experiment, alpha, gamma, epsilon, decay rate, episodes, use_random_values, action_space_steps, obs_space_steps]")

    for idx, hyperparameter in enumerate(hyperparameters):
        metrics = test_one_hyperparameter(hyperparameter)
        print(f"Hyperparameters {hyperparameter} ")
        print(f"Average reward: {metrics[0]:.2f}")
        print(f"Max reward: {metrics[1]:.2f}")
        print(f"Average steps: {metrics[2]:.2f}")
        print(f"Success rate: {metrics[3]:.2f}")
        print(f"---------------------{idx}/{len(hyperparameters)}------------------------")
