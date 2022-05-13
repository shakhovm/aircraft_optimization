import torch
import numpy as np
from aircraft_env import AircraftEnv
from itertools import product
import torch.nn.functional as F
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


class ActorCriticAgent(GeneralAgent):
    def __init__(self):
        ...

    def train_model(self):

        s, a, r, s_prime, done = self.make_batch()

        td_target = r + self.gamma * self.model.v(s_prime) * done
        delta = td_target - self.model.v(s)
        pi = self.model.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.model.v(s), td_target.detach())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()