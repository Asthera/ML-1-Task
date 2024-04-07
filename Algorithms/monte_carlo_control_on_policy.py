import numpy as np
import gymnasium as gym
import pickle


class MonteCarloOnPolicy:
    def __init__(self,
                 env: gym.wrappers,
                 gamma: float,
                 epsilon: float,
                 decay_rate: float,
                 use_random_values: bool,
                 observation_space_steps: [int, int],
                 action_space_steps: int):

        self.env = env

        # define arbitrary e-soft policy (random)
        self.pi = np.full((observation_space_steps[0], observation_space_steps[1], action_space_steps),
                          1.0 / action_space_steps)

        # define Q(s, a) arbitrarily
        if use_random_values:
            self.Q = np.random.rand(observation_space_steps[0], observation_space_steps[1], action_space_steps)
        else:
            self.Q = np.zeros((observation_space_steps[0], observation_space_steps[1], action_space_steps))

        # define Returns(s, a)
        self.Returns = {}

        # define hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.observation_space_steps = observation_space_steps
        self.action_space_steps = action_space_steps

        # define spaces
        self.action_space = np.linspace(-1.0, 1.0, action_space_steps)
        self.position_space = np.linspace(-1.2, 0.6, observation_space_steps[0])
        self.velocity_space = np.linspace(-0.07, 0.07, observation_space_steps[1])

        # steps/rewards to then plot it in the end
        self.steps = []
        self.rewards = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.action_space_steps))
        else:
            return np.argmax(self.pi[state[0], state[1], :])

    def discretize_state(self, state):
        position = np.digitize(state[0], self.position_space)
        velocity = np.digitize(state[1], self.velocity_space)
        return position, velocity

    def generate_episode(self):
        episode = []
        state, info = self.env.reset()
        state = self.discretize_state(state)

        done = False

        while not done and len(episode) < 100_000:
            # choose action using epsilon-greedy policy
            action = self.choose_action(state)

            # take action
            state_, reward, done, truncated, info = self.env.step([action])
            state_ = self.discretize_state(state_)
            episode.append((state, action, reward))
            state = state_

            # if len(episode) % 10000 == 0:
            #     print(f"Generated episode step {len(episode)}")

        # print("Generated episode length: ", len(episode), " steps.)")

        self.steps.append(len(episode))
        self.rewards.append(sum([r for (s, a, r) in episode]))

        return episode

    def train_on_episode(self, episode):
        G = 0
        episode_set = set([(s, a) for (s, a, r) in episode])

        # Here we iterate over the episode in reverse order
        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            # Calculate the return
            G = self.gamma * G + reward

            # If the state-action pair is unique in this episode (first visit)
            # then update the Q-value and policy
            if (state, action) not in episode_set:
                continue

            # Update Returns and Q-values
            if self.Returns.get((state, action)) is None:
                self.Returns[(state, action)] = []

            self.Returns[(state, action)].append(G)
            self.Q[state[0], state[1], action] = np.mean(self.Returns[(state, action)])

            # Policy improvement
            best_action = np.argmax(self.Q[state[0], state[1], :])

            for a in range(self.action_space_steps):
                if a == best_action:
                    self.pi[state[0], state[1], a] = (1 - self.epsilon) + (self.epsilon / self.action_space_steps)
                else:
                    self.pi[state[0], state[1], a] = self.epsilon / self.action_space_steps

            # if t % 2000 == 0:
            #     print(f"Episode step {t} done")

    def train(self, episodes):

        # we will use it for early stopping
        stop = 0

        for e in range(episodes):
            episode = self.generate_episode()
            self.train_on_episode(episode)
            self.decrease_epsilon()

            if len(episode) > 90_000:
                stop += 1
            else:
                stop = 0


            if stop > 4:
                break

            # print(f"Episode {e}/{episodes} done, steps: {len(episode)}")

    def decrease_epsilon(self):
        self.epsilon -= self.decay_rate

        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def save_weights(self, path: str):
        np.save(path + "_Q", self.Q)
        np.save(path + "_policy", self.pi)

        with open(path + "_returns" + ".pkl", 'wb') as file:
            pickle.dump(self.Returns, file)

    def load_weights(self, path: str):
        self.Q = np.load(path + "_Q" + ".npy")
        self.pi = np.load(path + "_policy" + ".npy")

        with open(path + "_returns" + ".pkl", 'rb') as file:
            self.Returns = pickle.load(file)

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

    # Train agent
    agent = MonteCarloOnPolicy(env, gamma, epsilon, decay_rate, use_random_values, obs_space_steps,
                               action_space_steps)
    agent.train(hyperparameter[5])
    agent.save_weights(f"weights/mc_on_policy/{hyperparameter[0]}")
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
