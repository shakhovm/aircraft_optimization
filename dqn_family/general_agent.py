from itertools import product

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from aircraft_env import AircraftEnv
from dqn_family.replay_buffer import ReplayBuffer
from utils.units_converter import feet2meter


def normilize_value(data, data_max, data_min):
    return (data - data_min) / (data_max - data_min)


def unnormilize_data(x, dmax, dmin):
    return x * (dmax - dmin) + dmin


class TrainingState:
    def __init__(self):
        self.loss = [0.]
        self.rewards = [0.]
        self.episode_rewards = [0.]
        self.avarage_fuel_burns = [0.]
        self.episode_fuel_burns = [0.]
        self.fuel_burn = 0
        self.steps_of_episode = 0
        self.episode_reward = 0
        self.dict_to_show = {
            "AvarageFuel": self.avarage_fuel_burns,
            "AverageReward": self.rewards,
            "Loss": self.loss,
            "EpisodeReward": self.episode_rewards,
            "EpisodeFuel": self.episode_fuel_burns
        }


class DQNFTrainingState(TrainingState):
    def __init__(self):
        super(DQNFTrainingState, self).__init__()
        self.frontier_loss = [0]
        self.loss_q = [0]


class GeneralAgent:
    def __init__(self,
                 config,
                 model_type,
                 env: AircraftEnv
                 ):

        if 'device' in config:
            self.device = config['device']
        else:
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

        self.training_state = TrainingState()

        # Environment

        self.actions = np.array(list(product(np.arange(env.n_routes),
                                             list(map(feet2meter, np.arange(-2000, 3000, 1000))),
                                                np.arange(env.cruise_mach_range[0],
                                                          env.cruise_mach_range[-1], 0.01))))
        # self.actions = np.array(list(map(feet2meter, np.arange(-2000, 3000, 1000))))
        self.env = env
        self.action_number = len(self.actions)
        print(f'Actions Number {self.action_number}')
        if "episode_init" in config:
            self.episode_init = config["episode_init"]
        else:
            self.episode_init = "begin"

        # Model

        # self.update_step = config['update_step']
        self.model = model_type(len(self.actions)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        if self.load_model:
            self.model.load_state_dict(torch.load(self.path_to_load))

    def preprocess_action(self, action):
        processed_action = self.actions[action]
        return {
            'trajectory': int(processed_action[0]),
            'altitude': processed_action[1],
            'mach_number': processed_action[2] #0.77
        }

    def preprocess_state(self, state):
        location = self.env.waypoints[state['trajectory']][state['waypoint']]
        return np.array([
            normilize_value(location.latitude, 90, -90),
            normilize_value(location.longitude, 180, -180),
            normilize_value(state['altitude'], self.env.cruise_alt_max, self.env.cruise_alt_min)
        ])

    def transition_process(self, o_state, o_act, o_reward, o_next_state, o_done):
        return \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_state)).to(self.device)), \
            torch.autograd.Variable(torch.LongTensor(o_act).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_reward).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_next_state)).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_done).to(self.device))

    def init_new_episode(self, env, mode="random"):
        if mode == "random":
            observation = env.random_state()
        else:
            observation = env.reset()
        self.state = self.preprocess_state(observation)

    def episode_end_unique(self, episode):
        pass

    def episode_check(self, episode):
        if (episode + 1) % self.episode_to_save == 0:
            torch.save(self.model.state_dict(), self.path_to_save)

        if (episode + 1) % self.episode_to_monitor == 0:
            # #
            # writer.add_scalar('Episode Reward', np.array(self.rewards).mean(), episode)
            # self.rewards = [self.rewards[-1]]
            #
            # #     marg = 0
            #
            # writer.add_scalar('Fuel', -np.array(self.fuel_burns).mean(), episode)
            # self.fuel_burns = [self.fuel_burns[-1]]
            # # self.writer.add_scalar('Reward', self.all_rewards[-1], episode)
            # self.all_rewards = [self.all_rewards[-1]]
            # #
            for key, value in self.training_state.dict_to_show.items():
                if len(value) > 0:
                    self.writer.add_scalar(key, np.array(value[-1]), episode)
                    self.training_state.dict_to_show[key].clear()
            self.writer.flush()

    def choose_action(self):
        raise NotImplementedError

    def on_training_begin(self):
        trangle = trange(self.episodes, desc='Training', leave=True)

        trangle.set_description("Began Training!")
        trangle.reset()
        self.writer = SummaryWriter()
        self.init_new_episode(self.env, self.episode_init)
        return trangle

    def on_episode_begin(self, env):
        action = self.choose_action()
        processed_action = self.preprocess_action(action)
        next_observation, reward, done = env.step(processed_action)
        next_state = self.preprocess_state(next_observation)
        return action, processed_action, next_state, reward, done, next_observation

    def on_episode_mid(self, action, processed_action, next_state, reward, done, next_observation):
        raise NotImplementedError

    def on_episode_end(self, episode):
        raise NotImplementedError

    def on_done(self, done):
        if done:
            self.init_new_episode(self.env, self.episode_init)
            self.training_state.rewards.append(self.training_state.episode_reward /
                                               self.training_state.steps_of_episode)
            self.training_state.avarage_fuel_burns.append(
                -self.training_state.fuel_burn / self.training_state.steps_of_episode)
            self.training_state.episode_fuel_burns.append(-self.training_state.fuel_burn)
            self.training_state.episode_rewards.append(self.training_state.episode_reward)
            self.training_state.fuel_burn = 0
            self.training_state.steps_of_episode = 0
            self.training_state.episode_reward = 0
            self.on_save_best_model()

    def on_training_state(self, dct):
        self.training_state.steps_of_episode += 1
        self.training_state.episode_reward += dct["reward"]
        self.training_state.fuel_burn += self.env.state_info['fuel_burn']

    def on_save_best_model(self):
        last_reward = self.training_state.episode_fuel_burns[-1]

        if last_reward > self.best_score and self.epsilon < 0.4:
            print(last_reward)
            torch.save(self.model.state_dict(), self.path_to_save + 'best')
            # torch.save(self.model, self.path_to_save + '_best')
            self.best_score = last_reward

    def on_trangle(self, episode, trangle):
        if episode % self.episode_to_monitor == 0:
            trangle.set_description(
                f"Episode: {episode}"
            )
            trangle.refresh()

    def train(self):
        self.best_score = -10000
        trangle = self.on_training_begin()
        for episode in trangle:
            self.on_trangle(episode, trangle)
            action, processed_action, next_state, reward, done, next_observation = self.on_episode_begin(self.env)
            self.on_episode_mid(action, processed_action, next_state, reward, done, next_observation)

            self.on_done(done)
            self.on_episode_end(episode)


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
        self.max_buffer_len = config['max_buffer_len']
        self.replay_buffer = ReplayBuffer(self.max_buffer_len)

        # Model
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
        unnormilized_altitude = unnormilize_data(self.state[2], self.env.cruise_alt_max, self.env.cruise_alt_min)
        if (1 - self.epsilon) <= np.random.random():
            k = np.where(self.env.valid_action(unnormilized_altitude,  self.actions[:, 1]))[0]
            self.action = np.random.choice(k)
            # print(self.actions[self.action])
        else:
            state = torch.autograd.Variable(torch.FloatTensor(self.state).to(self.device).unsqueeze(0))
            # self.model.eval()
            # with torch.no_grad():

            model_value = self.model(state)
            k = np.where(~self.env.valid_action(unnormilized_altitude, self.actions[:, 1]))[0]

            if len(k) > 0:

                model_value[0, k] = -1000.
            self.action = model_value.max(1)[1].item()
            # self.model.train()
        return self.action

    def train_model(self):
        o_state, o_act, o_reward, o_next_state, o_done = \
            self.transition_process(*self.replay_buffer.sample(self.minibatch))

        # o_reward = (o_reward - o_reward.mean()) / (o_reward.std() + 1e-7)
        q = self.model(o_state).gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next = self.target_model(o_next_state)

        # for i in range(len(o_next_state)):
        #     unnormilized_altitude = unnormilize_data(o_next_state[i, 2], self.env.cruise_alt_max, self.env.cruise_alt_min)
        #     k = np.where(~self.env.valid_action(unnormilized_altitude, self.actions))[0]
        #     if len(k) > 0:
        #         q_next[i, k] = -1000.
        y_hat = o_reward + self.gamma * q_next.max(1)[0] * (1 - o_done)
        loss = (q - y_hat.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_state.loss.append(loss.detach().numpy())

    def choose_action(self):
        return self.epsilon_greedy()

    def on_episode_mid(self, action, processed_action, next_state, reward, done, next_observation):
        self.on_training_state({"reward": reward})
        self.replay_buffer.push((self.state, action, reward, next_state, done))
        self.state = next_state.copy()
        if len(self.replay_buffer) >= self.begin_train:
            self.train_model()

    def on_episode_end(self, episode):
        self.reduce_epsilon(episode)
        self.episode_check(episode)
        if (episode + 1) % self.copy_step == 0:
            self.update_target()

    def on_trangle(self, episode, trangle):
        if episode % self.episode_to_monitor == 0:
            trangle.set_description(
                f"Episode: {episode} | Epsilon: {self.epsilon}"
            )
            trangle.refresh()

    def best_action(self, state):
        processed_state = self.preprocess_state(state)

        processed_state = torch.autograd.Variable(torch.FloatTensor(processed_state).to(self.device).unsqueeze(0))
        # print(self.model.state)
        model_value = self.model(processed_state)
        k = np.where(~self.env.valid_action(state['altitude'], self.actions[:, 1]))[0]

        if len(k) > 0:
            model_value[0, k] = -1000.
        action = model_value.max(1)[1].item()
        processed_action = self.preprocess_action(action)
        return processed_action
