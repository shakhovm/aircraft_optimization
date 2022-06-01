import torch
import numpy as np

from dqn_family.general_agent import DQNFamily
from ddqn_agent.ddqn_model import LinearDDQNModel
from ddqn_agent.dqn_model import Net

# class DDQNF_TraningState

from dqn_family.general_agent import unnormilize_data
class DDQNFAgent(DQNFamily):
    def __init__(self, conf, env):
        if conf['model'] == "dqnf":
            model_type = Net
        elif conf['model'] == 'ddqnf':
            model_type = LinearDDQNModel

        super().__init__(config=conf, env=env, model_type=model_type)
        self.margin = conf['margin']
        self.lamd = conf['lambda']

    def train_model(self):
        state, act, reward, next_state, done = \
            map(lambda x: np.array(x), self.replay_buffer.sample(self.minibatch))
        o_state, o_act, o_reward, o_next_state, o_done = self.transition_process(state, act, reward, next_state, done)
        o_reward = (o_reward - o_reward.mean()) / (o_reward.std() + 1e-7)
        # o_reward = torch.log(o_reward)
        q = self.model(o_state) #.gather(1, o_act.unsqueeze(1)).squeeze(1)

        q_next = self.target_model(o_next_state)
        y_hat = o_reward + self.gamma * q_next.max(1)[0] * (1 - o_done)
        loss_q = (q.gather(1, o_act.unsqueeze(1)).squeeze(1) - y_hat.detach()).pow(2).mean()
        loss_frontier = torch.Tensor([0])
        cpu_q = q #.to('cpu')
        for i in range(len(state)):
            # print(state[i, 2])
            unnormilized_altitude = unnormilize_data(state[i, 2], self.env.cruise_alt_max,
                                                     self.env.cruise_alt_min)
            if not self.env.valid_action(unnormilized_altitude, self.actions[act[i], 1]):

                #         print(state[i, 2], agent.actions[act[i]])
                k = np.where(self.env.valid_action(unnormilized_altitude, self.actions[:, 1]))[0]
                # .gather(0, torch.LongTensor(k, device='cpu'))#.squeeze(1).size()
                min_q_valid = cpu_q[i, k].min()
                q_invalid = cpu_q[i, act[i]] + self.margin
                #         print(q_invalid)
                if min_q_valid < q_invalid:
                    loss_frontier += (min_q_valid - q_invalid).pow(2)
        loss = loss_q + self.lamd*loss_frontier.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_state.loss.append(loss.item())
        self.training_state.loss.append(self.lamd*loss_frontier.item())
        self.training_state.loss.append(loss_q.item())


    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.random():
            self.action = np.random.randint(self.action_number)
            # print(self.actions[self.action])
        else:
            state = torch.autograd.Variable(torch.FloatTensor(self.state).to(self.device).unsqueeze(0))
            # self.model.eval()
            # with torch.no_grad():

            model_value = self.model(state)
            self.action = model_value.max(1)[1].item()
            # self.model.train()
        return self.action