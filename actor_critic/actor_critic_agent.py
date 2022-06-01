import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from actor_critic.actor_critic_model import ActorCritic
from dqn_family.general_agent import GeneralAgent
#

def unnormilize_data(x, dmax, dmin):
    return x * (dmax - dmin) + dmin


class ActorCriticAgent(GeneralAgent):
    def __init__(self, conf, env):
        super(ActorCriticAgent, self).__init__(conf, env=env, model_type=ActorCritic)
        self.data = []
        self.minibatch = conf['minibatch']

    def train_model(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.model.v(s_prime) * done
        value = self.model.v(s)
        delta = td_target - value

        # invalid_actions = np.where(~self.env.valid_action(unnormilized_altitude, self.actions[:, 1]))[0]
        # out = self.model.pi(torch.from_numpy(self.state).float())
        pi = self.model.pi(s, softmax_dim=1)
        # for i in range(len(s)):
        #     unnormilized_altitude = unnormilize_data(s[i , 2], self.env.cruise_alt_max, self.env.cruise_alt_min)
        #     invalid_actions = np.where(~self.env.valid_action(unnormilized_altitude, self.actions[:, 1]))[0]
        #     pi[i][invalid_actions] = -1000.
        #     # print(pi[i])
        pi = F.softmax(pi, dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(value, td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.training_state.loss.append(loss.detach().mean().numpy())

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def choose_action(self):
        # trangle.refresh()
        unnormilized_altitude = unnormilize_data(self.state[2], self.env.cruise_alt_max, self.env.cruise_alt_min)
        invalid_actions = np.where(~self.env.valid_action(unnormilized_altitude, self.actions))[0]
        out = self.model.pi(torch.from_numpy(self.state).float())
        # print(out)
        out[invalid_actions] = -10000
        # print(out)
        prob = F.softmax(out, dim=0)
        # print(prob)
        m = Categorical(prob)
        # print(prob, m)
        action = m.sample()
        return action

    def on_episode_mid(self, action, processed_action, next_state, reward, done, next_observation):
        self.on_training_state({"reward": reward})
        self.put_data((self.state, action, reward, next_state, done))
        self.state = next_state.copy()
        # if len(self.replay_buffer) >= self.begin_train:
        #     self.train_model()

    def on_episode_end(self, episode):
        if (episode + 1) % self.minibatch == 0:
            self.train_model()
        self.episode_check(episode)

    def best_action(self, state):
        processed_state = self.preprocess_state(state)

        invalid_actions = np.where(~self.env.valid_action(state['altitude'], self.actions[:, 1]))[0]
        prob = self.model.pi(torch.from_numpy(processed_state).float())
        print(F.softmax(prob, 0))
        prob[invalid_actions] = -1000.
        prob = F.softmax(prob, 0)

        print(prob)
        m = Categorical(prob)
        # print(prob, m)
        action = m.sample()
        # print(action)
        processed_action = self.preprocess_action(action)
        return processed_action
