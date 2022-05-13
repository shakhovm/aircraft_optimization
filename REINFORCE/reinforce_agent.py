import torch
from tqdm import trange
from REINFORCE.reinforce_model import REINFORCE
import numpy as np
from aircraft_env import AircraftEnv
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from utils.units_converter import feet2meter
from torch.distributions import Categorical


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
        self.model = model_type(len(self.actions))#.to(self.device)
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


class REINFORCEAgent(GeneralAgent):
    def __init__(self,
                 config,
                 env: AircraftEnv
                 ):

        super().__init__(config,
                         model_type=REINFORCE,
                         env=env
                         )

        # Main Params

        self.minibatch = config['minibatch']

        # Episode Params

        self.begin_train = config['begin_train']
        self.copy_step = config['copy_step']
        self.episode_steps = config['episode_steps']

        # Model Fields

        self.action = None
        self.state = None
        # self.replay_buffer = ReplayBuffer(config['max_buffer_len'])
        self.data = []

        # Model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        # self.target_model = model_type(self.action_number).to(self.device)
        # self.update_target()

        # Rewards

        self.rewards = []
        self.losses = []
        self.periodic_reward = 0
        self.periodic_rewards = []
    #
    # def update_target(self):
    #     self.target_model.load_state_dict(self.model.state_dict())

    def init_new_episode(self, env):
        observation = env.reset()
        self.state = self.preprocess_state(observation)

    def episode_check(self, episode):

        # if (episode + 1) % self.copy_step == 0:
        #     # self.losses.append(loss)
        #     self.update_target()

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

    def one_iter(self, reward, log_prob, R):

        return R

    def train_model(self):
        R = 0
        loss = 0
        R = 0
        policy_loss = []
        returns = []
        log_probs = []
        for r, log_prob in self.data[::-1]:
            R = r + self.gamma * R
            log_probs.append(log_prob)
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        print(policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        del self.data

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
        self.rewards.append(-10000)
        self.fuel_burns.append(10000)
        self.training_state = {
            "loss": [0]
        }
        fuel_burn = 0
        steps_of_episode = 0
        for episode in trangle:
            if episode % self.episode_to_monitor == 0:
                trangle.set_description(
                    f"Episode: {episode}"
                )
            trangle.refresh()
            prob = self.model.act(torch.from_numpy(self.state).float())
            m = Categorical(prob)
            print(prob, m)
            action = m.sample()
            print(action)
            processed_action = self.preprocess_action(action)
            next_observation, reward, done = env.step(processed_action)

            steps_of_episode += 1
            total_reward += reward
            episode_reward += reward
            fuel_burn += env.state_info['fuel_burn']
            self.all_rewards.append(reward)
            self.periodic_reward += reward
            next_state = self.preprocess_state(next_observation)
            self.data.append((reward, torch.log(prob[action])))
            self.state = next_state.copy()


            if done:
                self.train_model()
                self.init_new_episode(env)
                self.rewards.append(episode_reward / steps_of_episode)
                self.fuel_burns.append(fuel_burn / steps_of_episode)

                fuel_burn = 0
                episode_reward = 0
                steps_of_episode = 0

            self.episode_check(episode)

    # def best_action(self, state):
    #     state = self.preprocess_state(state)
    #
    #     state = torch.autograd.Variable(torch.FloatTensor(state).to(self.device).unsqueeze(0))
    #     # print(self.model.state)
    #     action = self.target_model(state).max(1)[1].item()
    #
    #     processed_action = self.preprocess_action(action)
    #     return processed_action

    def w_release(self):
        self.writer.close()
