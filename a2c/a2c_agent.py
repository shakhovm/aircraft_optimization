import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical

from actor_critic.actor_critic_agent import unnormilize_data
from utils.geodesic import Location
from actor_critic.actor_critic_model import ActorCritic
from dqn_family.general_agent import GeneralAgent
from aircraft_env import AircraftEnv
import numpy as np
import torch

def normilize_value(data, data_max, data_min):
    return (data - data_min) / (data_max - data_min)


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    loc_1 = Location(45.46873715, -73.74257166095532)
    loc_2 = Location(49.0068908, 2.5710819691019156)
    env = AircraftEnv(arrival_location=loc_1, destination=loc_2, n_waypoints=9)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        # elif cmd == 'reset_task':
        #     ob = env.reset_task()
        #     worker_end.send(ob)
        # elif cmd == 'close':
        #     worker_end.close()
        #     break
        # elif cmd == 'get_spaces':
        #     worker_end.send((env.observation_space, env.action_space))
        # else:
        #     raise NotImplementedError


class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        loc_1 = Location(45.46873715, -73.74257166095532)
        loc_2 = Location(49.0068908, 2.5710819691019156)
        self.env = AircraftEnv(arrival_location=loc_1, destination=loc_2, n_waypoints=9)
        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    @property
    def waypoints(self):
        return self.env.waypoints

    @property
    def cruise_alt_min(self):
        return self.env.cruise_alt_min

    @property
    def cruise_alt_max(self):
        return self.env.cruise_alt_max

    def valid_action(self, altitude, action):
        return self.env.valid_action(altitude, action)

    def valid_action2(self, altitude, wp, trajectory):
        self.env.valid_action2(altitude, wp, trajectory)

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def preprocess_state(self, recv):
        if not isinstance(recv, dict):
            state = recv[0]
            location = self.env.waypoints[state['trajectory']][state['waypoint']]
            return np.array([
                normilize_value(location.latitude, 90, -90),
                normilize_value(location.longitude, 180, -180),
                normilize_value(state['altitude'], self.env.cruise_alt_max, self.env.cruise_alt_min)
            ]), recv[1], recv[2]
        else:
            state = recv
            location = self.env.waypoints[state['trajectory']][state['waypoint']]
            return np.array([
                normilize_value(location.latitude, 90, -90),
                normilize_value(location.longitude, 180, -180),
                normilize_value(state['altitude'], self.env.cruise_alt_max, self.env.cruise_alt_min)
            ])

    def step_wait(self):
        results = [self.preprocess_state(master_end.recv()) for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        results = [self.preprocess_state(master_end.recv()) for master_end in self.master_ends]
        # l = np.stack([master_end.recv() for master_end in self.master_ends])
        # print(l)
        return np.stack(results)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        # for master_end in self.master_ends:
        #     master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True


class A2CAgent(GeneralAgent):
    def __init__(self, conf, env):
        super(A2CAgent, self).__init__(conf, env=env, model_type=ActorCritic)
        self.data = []
        self.minibatch = conf['minibatch']
        self.worker_num = conf['worker_number']
        self.env = ParallelEnv(self.worker_num)
        self.s_lst, self.a_lst, self.r_lst, self.mask_lst = list(), list(), list(), list()

    def compute_target(self, v_final, r_lst, mask_lst):
        G = v_final.reshape(-1)
        td_target = list()
        for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
            G = r + self.gamma * G * mask
            td_target.append(G)
        return torch.tensor(td_target[::-1]).float()

    def preprocess_state(self, state):
        return state

    def train_model(self):

        s_final = torch.from_numpy(self.state).float()
        v_final = self.model.v(s_final).detach().clone().numpy()
        td_target = self.compute_target(v_final, self.r_lst, self.mask_lst)

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(self.s_lst).float().reshape(-1, 3)  # 4 == Dimension of state
        a_vec = torch.tensor(self.a_lst).reshape(-1).unsqueeze(1)
        mod = self.model.v(s_vec)
        advantage = td_target_vec - mod.reshape(-1)

        pi = F.softmax(self.model.pi(s_vec, softmax_dim=1), 1)
        pi_a = pi.gather(1, a_vec).reshape(-1) + 1e-7
        loss = -(torch.log(pi_a) * advantage.detach()).mean() + \
               F.smooth_l1_loss(self.model.v(s_vec).reshape(-1), td_target_vec)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_state.loss.append(loss.detach().numpy())

    def choose_action(self):
        # trangle.refresh()
        # prob = self.model.pi(torch.from_numpy(self.state).float())
        # m = Categorical(prob)
        # print(prob, m)
        out = self.model.pi(torch.from_numpy(self.state).float())
        for i in range(self.worker_num):
            unnormilized_altitude = unnormilize_data(self.state[i, 2], self.env.cruise_alt_max, self.env.cruise_alt_min)
            invalid_actions = np.where(~self.env.valid_action(unnormilized_altitude, self.actions))[0]

            # print(out)
            out[i][invalid_actions] = -100
            # print(out)
        # print(out)
        prob = F.softmax(out, dim=1)
        m = Categorical(prob)
        action = m.sample().numpy()
        return action

    def on_episode_begin(self, env):
        actions = self.choose_action()
        processed_action = [self.preprocess_action(action) for action in actions]
        next_observation, reward, done = env.step(processed_action)
        next_state = self.preprocess_state(next_observation)
        return actions, processed_action, next_state, reward, done, next_observation

    def on_episode_mid(self, action, processed_action, next_state, reward, done, next_observation):
        self.s_lst.append(self.state)
        self.a_lst.append(action)
        self.r_lst.append(reward / 100.0)
        self.mask_lst.append(1 - done)
        self.state = next_state.copy()

    def on_done(self, done):
        pass

    def on_episode_end(self, episode):
        if (episode + 1) % self.minibatch == 0:

            self.train_model()
            self.s_lst, self.a_lst, self.r_lst, self.mask_lst = list(), list(), list(), list()
        if (episode + 1) % self.episode_to_monitor == 0:
            loc_1 = Location(45.46873715, -73.74257166095532)
            loc_2 = Location(49.0068908, 2.5710819691019156)
            env = AircraftEnv(arrival_location=loc_1, destination=loc_2, n_waypoints=9)
            score = 0.0
            done = False
            num_test = 10

            # for _ in range(num_test):
            s = env.reset()
            s = self.env.preprocess_state(s)
            fuel_burn = 0
            while not done:
                prob = self.model.pi(torch.from_numpy(s).float(), softmax_dim=0)
                unnormilized_altitude = unnormilize_data(s[2], self.env.cruise_alt_max,
                                                         self.env.cruise_alt_min)
                invalid_actions = np.where(~self.env.valid_action(unnormilized_altitude, self.actions))[0]

                # print(out)
                prob[invalid_actions] = -100
                prob = F.softmax(prob, dim=0)
                a = Categorical(prob).sample().numpy()
                a = self.preprocess_action(a)
                s_prime, r, done = env.step(a)
                s = self.env.preprocess_state(s_prime)
                score += r
                fuel_burn -= env.state_info['fuel_burn']
            self.training_state.episode_rewards.append(score)
            self.training_state.episode_fuel_burns.append(fuel_burn)

            # done = False

        self.episode_check(episode)

    def best_action(self, state, env, sample='max'):
        processed_state = self.env.preprocess_state(state)
        iss = []
        ks = []
        # print('hello')
        # print(len(self.actions))
        # print(state['altitude'] + self.actions[100][1], state['waypoint'], int(self.actions[100][0]))
        for k in range(len(self.actions)):
            if not env.valid_action2(state['altitude'] + self.actions[k][1], state['waypoint'], int(self.actions[k][0])):
                # iss.append(i)
                # print(state['altitude'] + self.actions[k][1])
                ks.append(k)
        # print(ks)
        # invalid_actions = np.where(~self.env.valid_action2(state['altitude'] +, self.actions))[0]
        prob = self.model.pi(torch.from_numpy(processed_state).float())
        prob[np.array(ks)] = -10000.
        prob = F.softmax(prob, 0)

        if sample == 'max':
            action = prob.argmax().item()
        else:
            # print(prob)
            m = Categorical(prob)
            # print(prob, m)
            action = m.sample()

        processed_action = self.preprocess_action(action)
        return processed_action

