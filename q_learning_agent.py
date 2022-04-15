import numpy as np
from tqdm import trange

from aircraft_env import AircraftEnv, feet2meter
from functools import reduce


class GeneralAgent:
    def __init__(self,
                 env: AircraftEnv,
                 episode_number=1000,
                 learning_rate=1.0,
                 discount_factor=0.0,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.9,
                 alt_delta=1000 * 0.3048,
                 mach_delta=0.01
                 ):
        self.episode_number = episode_number
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None
        self._num_actions = 3

        # State = (Waypoint, Altitude)
        self.env = env


class QTableAgent(GeneralAgent):
    def __init__(self,
                 env: AircraftEnv,
                 episode_number=5000,
                 learning_rate=0.95,
                 save_dir='',
                 checkpoint='',
                 discount_factor=0.9,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.9,
                 alt_delta=1000 * 0.3048,
                 mach_delta=0.01,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_delta=3000
                 ):

        super().__init__(env, episode_number, learning_rate, discount_factor, exploration_rate,
                         exploration_decay_rate, alt_delta, mach_delta)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_delta = epsilon_delta
        self.epsilon = epsilon_start
        self.save_dir = save_dir

        self.trangle = trange(self.episode_number, desc='Training', leave=True)
        self.states = [
            np.arange(env.n_routes),
            np.arange(env.n_waypoints),
            np.arange(env.cruise_alt_min, env.cruise_alt_max + alt_delta, alt_delta),
        ]

        self.actions = [
            np.arange(env.n_routes),
            np.array(list(map(feet2meter, np.arange(-2000, 3000, 1000)))),
            np.arange(env.cruise_mach_range[0], env.cruise_mach_range[-1], mach_delta)
        ]
        if checkpoint:
            with open(checkpoint, 'rb') as f:
                self.q = np.load(f)
        else:
            self.q = np.zeros((self.states[0].shape[0], self.states[1].shape[0],
                               self.states[2].shape[0],
                               self.actions[0].shape[0], self.actions[1].shape[0],
                               self.actions[2].shape[0])) #- 1060e3
        self._print_summary()

    def _print_summary(self):
        info = f'States (Trajectory, Waypoints, Altitude) ' \
               f'{" x ".join([str(x.shape[0]) for x in self.states])}' \
               f' / {np.prod([x.shape[0] for x in self.states])}\n' \
               f'Action (Trajectory, Altitude Step, Mach Number) ' \
               f'{" x ".join([str(x.shape[0]) for x in self.actions])} / ' \
               f'{np.prod([x.shape[0] for x in self.states])}\n' \
               f'Q Table {self.q.shape} / {np.prod(self.q.shape)}'
        print(info)

    def max_action(self, state):
        q_state = tuple(state)
        action = np.array(np.unravel_index(self.q[q_state].argmax(), self.q[q_state].shape))
        return action

    def epsilon_greedy(self):
        un = np.random.uniform(0, 1)
        enable_exploration = un <= self.epsilon
        if enable_exploration:
            action_1 = np.random.randint(len(self.actions[0]))
            action_2 = np.random.randint(len(self.actions[1]))
            action_3 = np.random.randint(len(self.actions[2]))
            action = np.array([action_1, action_2, action_3])
        else:
            action = self.max_action(self.state)
        return action

    def reduce_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * episode / self.epsilon_delta)

    def preprocess_state(self, state):

        return np.array([
            state["trajectory"],
            state['waypoint'],
            np.digitize(state['altitude'], self.states[2]) - 1
        ])

    def preprocess_action(self, action):
        return {
            "trajectory": self.actions[0][action[0]],
            "altitude": self.actions[1][action[1]],
            "mach_number": self.actions[2][action[2]]
        }

    def act(self, new_state, reward):
        next_state = self.preprocess_state(new_state)
        q_current_state_action = tuple(np.concatenate((self.state, self.action)))
        next_q_max = self.q[tuple(next_state)].max()
        current_q = self.q[q_current_state_action]

        self.q[q_current_state_action] = current_q + self.learning_rate * \
                                         (reward + self.discount_factor * next_q_max - current_q)
        self.state = next_state

    def run_episode(self):
        self.state = self.preprocess_state(self.env.reset())
        done = False
        total_reward = 0
        while not done:
            self.action = self.epsilon_greedy()
            next_state, reward, done = self.env.step(self.preprocess_action(self.action))
            self.act(next_state, 1e6 + reward)
            total_reward += reward
        return total_reward

    def train(self):
        self.rewards = []
        for episode in self.trangle:
            self.trangle.refresh()

            total_reward = self.run_episode()
            self.rewards.append(total_reward)
            # self.total_reward(total_reward, episode)
            self.trangle.set_description(
                f"Episode: {episode} | Episode Reward {total_reward} | Epsilone {self.epsilon}"
            )

            self.reduce_epsilon(episode)

        self.rewards = np.array(self.rewards)
        if self.save_dir:
            with open(self.save_dir, 'wb') as f:
                np.save(f, self.q)

    def best_action(self, state):
        return self.preprocess_action(self.max_action(self.preprocess_state(state)))
