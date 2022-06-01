from dqn_family.general_agent import DQNFamily
from double_dqn.double_dqn_model import DoubleDQNModel
import numpy as np
import torch


def unnormilize_data(x, dmax, dmin):
    return x * (dmax - dmin) + dmin


class DoubleDQNAgent(DQNFamily):
    def __init__(self, conf, env):
        super(DoubleDQNAgent, self).__init__(config=conf, env=env, model_type=DoubleDQNModel)

    def train_model(self):
        o_state, o_act, o_reward, o_next_state, o_done = \
            self.transition_process(*self.replay_buffer.sample(self.minibatch))

        # o_reward = (o_reward - o_reward.mean()) / (o_reward.std() + 1e-7)
        q = self.model(o_state).gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next_current = self.model(o_next_state)  # .gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next_target = self.target_model(o_next_state).gather(1, q_next_current.max(1)[1].unsqueeze(1)).squeeze(1)
        y_hat = o_reward + self.gamma * q_next_target * (1 - o_done)
        loss = (q - y_hat.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_state.loss.append(loss.detach().numpy())


class DoubleDqnF(DQNFamily):
    def __init__(self, conf, env):
        super(DoubleDqnF, self).__init__(config=conf, env=env, model_type=DoubleDQNModel)
        self.margin = conf['margin']
        self.lamd = conf['lambda']
        self.training_state.loss_q = []
        self.training_state.loss_frontier = []
        self.training_state.dict_to_show['LossQ'] = self.training_state.loss_q
        self.training_state.dict_to_show['LossFrontier'] = self.training_state.loss_frontier

    def train_model(self):
        o_state, o_act, o_reward, o_next_state, o_done = \
            self.transition_process(*self.replay_buffer.sample(self.minibatch))

        # o_reward = (o_reward - o_reward.mean()) / (o_reward.std() + 1e-7)
        q = self.model(o_state)
        q_next_current = self.model(o_next_state)  # .gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next_target = self.target_model(o_next_state).gather(1, q_next_current.max(1)[1].unsqueeze(1)).squeeze(1)
        y_hat = o_reward + self.gamma * q_next_target * (1 - o_done)
        loss_q = (q.gather(1, o_act.unsqueeze(1)).squeeze(1) - y_hat.detach()).pow(2).mean()
        loss_frontier = torch.Tensor([0])
        cpu_q = q.to('cpu')
        for i in range(len(o_state)):
            unnormilize_state = unnormilize_data(o_state[i, 2], self.env.cruise_alt_max, self.env.cruise_alt_min)

            if not self.env.valid_action(unnormilize_state, self.actions[o_act[i]]):
                #         print(state[i, 2], agent.actions[act[i]])
                k = np.where(self.env.valid_action(unnormilize_state, self.actions))[0]
                # .gather(0, torch.LongTensor(k, device='cpu'))#.squeeze(1).size()
                min_q_valid = cpu_q[i, k].min()
                q_invalid = cpu_q[i, o_act[i]] + self.margin
                #         print(q_invalid)
                if min_q_valid < q_invalid:
                    loss_frontier += (min_q_valid - q_invalid).pow(2)

        loss = self.lamd * loss_frontier + loss_q

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_state.loss_q.append(loss_q.detach().numpy())
        self.training_state.loss_frontier.append(self.lamd * loss_frontier.detach().numpy())
        self.training_state.loss.append(loss.detach().numpy())
