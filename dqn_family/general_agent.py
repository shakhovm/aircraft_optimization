import torch
from tqdm import trange
from dqn_family.replay_buffer import ReplayBuffer
import numpy as np
from aircraft_env import AircraftEnv
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from utils.units_converter import feet2meter


class GeneralAgent:
    def __init__(self,
                 config,
                 model_type,
                 env: AircraftEnv
                 ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.episodes = config['episodes']

        self.gamma = config['gamma']

        # I/O Params

        self.path_to_load = config['path_to_load']
        self.path_to_save = config['path_to_save']
        self.load_model = config['load_model']
        self.episode_to_save = config['episode_to_save']
        self.episode_to_monitor = config['episode_to_monitor']

        # Metrics

        self.rewards = []

        # Actions

        self.actions = np.array(list(product(np.arange(env.n_routes),
                                             list(map(feet2meter, np.arange(-2000, 3000, 1000))))))
        # np.arange(env.cruise_mach_range[0],
        #           env.cruise_mach_range[-1], 0.01))))

        self.env = env
        self.action_number = len(self.actions)
        print(f'Actions Number {self.action_number}')
        # Model

        # self.update_step = config['update_step']
        self.model = model_type(len(self.actions)).to(self.device)
        if self.load_model:
            self.model.load_state_dict(torch.load(self.path_to_load))

    def preprocess_action(self, action):
        processed_action = self.actions[action]

        return {
            'trajectory': int(processed_action[0]),
            'altitude': processed_action[1],
            'mach_number': 0.77  # processed_action[2]
        }

    def preprocess_state(self, state):
        location = self.env.waypoints[state['trajectory']][state['waypoint']]
        return np.array([
            location.latitude,
            location.longitude,
            state['altitude']
        ])

    def transition_process(self, o_state, o_act, o_reward, o_next_state, o_done):
        return \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_state)).to(self.device)), \
            torch.autograd.Variable(torch.LongTensor(o_act).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_reward).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_next_state)).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_done).to(self.device))


class DQNFamily(GeneralAgent):
    def __init__(self,
                 config,
                 model_type,
                 env: AircraftEnv
                 ):

        super().__init__(config,
                         model_type=model_type,
                         env=env
                         )
        # Epsilon

        self.epsilon_delta = config['epsilon_delta']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_start = config['epsilon_start']
        self.epsilon = config['epsilon_start']

        # Main Params

        self.minibatch = config['minibatch']

        # Episode Params

        self.begin_train = config['begin_train']
        self.copy_step = config['copy_step']
        self.episode_steps = config['episode_steps']

        # Model Fields

        self.action = None
        self.state = None
        self.replay_buffer = ReplayBuffer(config['max_buffer_len'])

        # Model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.target_model = model_type(self.action_number).to(self.device)
        self.update_target()

        # Rewards

        self.rewards = []
        self.losses = []
        self.periodic_reward = 0
        self.periodic_rewards = []

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reduce_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * episode / self.epsilon_delta)

    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.random():
            # k = np.where(self.env.valid_action(self.state[2], self.actions[:, 1]))[0]
            self.action = np.random.randint(self.action_number)
        else:
            state = torch.autograd.Variable(torch.FloatTensor(self.state).to(self.device).unsqueeze(0))
            # self.model.eval()
            # with torch.no_grad():
            self.action = self.model(state).max(1)[1].item()
            # self.model.train()
        return self.action

    def init_new_episode(self, env):
        observation = env.random_state()
        self.state = self.preprocess_state(observation)

    def episode_check(self, episode):

        if (episode + 1) % self.copy_step == 0:
            # self.losses.append(loss)
            self.update_target()

        # if episode % self.episode_steps == 0:
        #     self.periodic_rewards.append(self.periodic_reward / self.episode_steps)
        #     self.periodic_reward = 0
        if (episode + 1) % self.episode_to_save == 0:
            torch.save(self.model.state_dict(), self.path_to_save)

        if (episode + 1) % self.episode_to_monitor == 0:
            #
            self.writer.add_scalar('Episode Reward', np.array(self.rewards).mean(), episode)
            self.rewards = [self.rewards[-1]]

            #     marg = 0

            self.writer.add_scalar('Fuel', -np.array(self.fuel_burns).mean(), episode)
            self.fuel_burns = [self.fuel_burns[-1]]
            # self.writer.add_scalar('Reward', self.all_rewards[-1], episode)
            self.all_rewards = [self.all_rewards[-1]]
            #
            for key, value in self.training_state.items():
                self.writer.add_scalar(key, np.array(value[-1]).mean(), episode)
                self.training_state[key] = [value[-1]]

    def train_model(self):
        o_state, o_act, o_reward, o_next_state, o_done = \
            self.transition_process(*self.replay_buffer.sample(self.minibatch))

        # o_reward = (o_reward - o_reward.mean()) / (o_reward.std() + 1e-7)
        q = self.model(o_state).gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next = self.target_model(o_next_state)
        y_hat = o_reward + self.gamma * q_next.max(1)[0] * (1 - o_done)
        loss = (q - y_hat.detach()).pow(2).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return loss

    def train(self):
        trangle = trange(self.episodes, desc='Training', leave=True)

        trangle.set_description("Began Training!")
        trangle.reset()
        self.writer = SummaryWriter()
        env = self.env
        self.all_rewards = []
        self.fuel_burns = []
        self.init_new_episode(env)
        total_reward = 0
        episode_reward = 0
        loss = torch.Tensor([0])
        self.rewards.append(-10000)
        self.fuel_burns.append(10000)
        self.training_state = {
            "fronts": [0],
            "loss_q": [0],
            "loss": [0]
        }
        fuel_burn = 0
        steps_of_episode = 0
        episode_step = 0
        for episode in trangle:
            if episode % self.episode_to_monitor == 0:
                trangle.set_description(
                    f"Episode: {episode}"
                    f"| Epsilone: {round(self.epsilon, 3)}"
                )
            trangle.refresh()
            action = self.epsilon_greedy()
            processed_action = self.preprocess_action(action)
            next_observation, reward, done = env.step(processed_action)
            steps_of_episode += 1
            total_reward += reward
            episode_reward += reward
            fuel_burn += env.state_info['fuel_burn']
            self.all_rewards.append(reward)
            self.periodic_reward += reward
            next_state = self.preprocess_state(next_observation)
            self.replay_buffer.push((self.state, action, reward, next_state, done))
            self.state = next_state.copy()
            if len(self.replay_buffer) >= self.begin_train:
                self.train_model()

            if done:
                self.init_new_episode(env)
                self.rewards.append(episode_reward / steps_of_episode)
                self.fuel_burns.append(fuel_burn / steps_of_episode)

                fuel_burn = 0
                episode_reward = 0
                steps_of_episode = 0

            self.reduce_epsilon(episode)
            self.episode_check(episode)

    def best_action(self, state):
        state = self.preprocess_state(state)

        state = torch.autograd.Variable(torch.FloatTensor(state).to(self.device).unsqueeze(0))
        # print(self.model.state)
        action = self.target_model(state).max(1)[1].item()

        processed_action = self.preprocess_action(action)
        return processed_action

    def w_release(self):
        self.writer.close()
