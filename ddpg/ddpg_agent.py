from tqdm import trange

from dqn_family.replay_buffer import ReplayBuffer
from ddpg.actor_model import ActorModel
from ddpg.critic_model import CriticModel
import torch.nn.functional as F
import torch
from ddpg.ou_noise import OUNoise
import numpy as np


class DDPGAgent:
    def __init__(
            self,
            config,
            env
    ):
        # Env

        self.env = env

        # Model Creation

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic_model = CriticModel(config['alpha_critic'],
                                        config['tau'], 3, 3).to(self.device)

        self.actor_model = ActorModel(config['alpha_actor'],
                                      config['tau'],
                                      3, 3).to(self.device)
        if config['load_models']:
            self.actor_model.load_state_dict(torch.load(config['load_path_actor'])).to(self.device)
            self.critic_model.load_state_dict(torch.load(config['load_path_critic'])).to(self.device)

        self.critic_model_target = CriticModel(config['alpha_critic'], config['tau'],
                                               3, 3).to(self.device)
        self.critic_model_target.load_state_dict(self.critic_model.state_dict())
        self.actor_model_target = ActorModel(config['alpha_actor'],
                                             config['tau'],
                                             3, 3).to(self.device)

        self.actor_model_target.load_state_dict(self.actor_model.state_dict())

        # Params definitions
        self.episodes = config['episodes']
        self.train_begin = config['train_begin']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.epochs = config['epochs']
        self.episodes_to_train = config['episodes_to_train']
        self.best_reward = -10e5

        # Output params
        self.trangle = trange(self.episodes, desc='Training', leave=True)
        self.trangle.reset()
        self.episodes_to_save = config['episodes_to_save']
        self.path = config['path']
        self.reward_path = config['reward_path']

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(config['buffer_size'])

        # Noise
        self.noise = OUNoise(3)


    def preprocess_action(self, detached_action):
        return {
            'trajectory': int(round(2 * detached_action[0] + 2)),
            'altitude': 609.6 * detached_action[1],
            'mach_number': 0.05 * detached_action[2] + 0.75
        }

    def preprocess_state(self, state):
        location = self.env.waypoints[state['trajectory']][state['waypoint']]
        return np.array([
            location.latitude,
            location.longitude,
            state['altitude']
        ])

    def random_action(self):
        return torch.autograd.Variable(torch.FloatTensor(np.float32(
            np.random.uniform(-1, 1, 3)
        )).to(self.device))

    def transition_process(self, o_state, o_act, o_reward, o_next_state, o_done):

        return \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_state)).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_act).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_reward).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_next_state)).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_done).to(self.device))

    def train_models(self):
        for epoch in range(self.epochs):
            # Sampling and defining y and y hat
            state, action, reward, state_next, done = \
                self.transition_process(*self.replay_buffer.sample(self.batch_size))
            y_hat = reward + self.gamma * self.critic_model_target(state_next,
                                                                   self.actor_model_target(state_next)) * (1 - done)

            y = self.critic_model(state, action)
            # Critic train
            critic_loss = F.smooth_l1_loss(y, y_hat.detach())
            self.critic_model.optimizer.zero_grad()
            critic_loss.backward()
            self.critic_model.optimizer.step()

            # Actor train
            actor_loss = -self.critic_model(state, self.actor_model(state)).mean()  # As we have gradient descent
            self.actor_model.optimizer.zero_grad()
            actor_loss.backward()
            self.actor_model.optimizer.step()

            # Update weights

            self.critic_model_target.update_weights(self.critic_model.parameters())
            self.actor_model_target.update_weights(self.actor_model.parameters())
        return actor_loss, critic_loss

    def train(self):
        self.rewards = [-1]
        state = self.env.reset()
        state = self.preprocess_state(state)
        episode_reward = 0
        actor_loss, critic_loss = 1000, 1000
        processed_action = {}
        for episode in self.trangle:

            rounded_rp_rew = str(round(self.rewards[-1]))
            if len(rounded_rp_rew) < 6:
                rounded_rp_rew = rounded_rp_rew[0] + (6 - len(rounded_rp_rew)) * " " + rounded_rp_rew[1:]
            self.trangle.set_description(
                f"Episode: {episode} | Episode Reward {rounded_rp_rew} |"
                f"Action: {processed_action}"

            )
            #f"Actor Loss: {actor_loss} | Critic Loss: {critic_loss}"
            self.trangle.refresh()
            if len(self.replay_buffer) < self.train_begin:
                action = self.random_action()
            else:
                action = self.actor_model(torch.autograd.Variable(torch.FloatTensor(state).to(self.device)))

            detached_action = action.detach().cpu().numpy()
            detached_action += self.noise.sample()

            detached_action = np.clip(detached_action, -1 , 1)
            processed_action = self.preprocess_action(detached_action)

            state_next, reward, done = self.env.step(processed_action)
            state_next = self.preprocess_state(state_next)
            self.replay_buffer.push((state, detached_action, reward,
                                     state_next, done))
            episode_reward += reward
            state = state_next
            if done:
                self.rewards.append(episode_reward)
                episode_reward = 0
                state = self.env.reset()
                state = self.preprocess_state(state)

            if len(self.replay_buffer) >= self.train_begin and ((episode + 1) % self.episodes_to_train == 0):
                actor_loss, critic_loss = self.train_models()

            # if (episode + 1) % self.episodes_to_print == 0:
            #     print(f"For episode {episode + 1} score is {score / self.episodes_to_print}")
            #     print(f"Critic Loss is {actor_loss}, Actor Loss is {critic_loss}")
            #     if score > self.best_reward:
            #         self.best_reward = score
            #         self.save_model(self.path + '_best')
            #
            #     score = 0

            # if (episode + 1) % self.episodes_to_save == 0:
            #     self.save_model(self.path)
        self.trangle.close()

    def save_model(self, path):
        torch.save(self.actor_model_target.state_dict(), path + "_actor_model")
        torch.save(self.critic_model_target.state_dict(), path + "_critic_model")
