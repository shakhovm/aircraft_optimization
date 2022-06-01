from dqn_family.general_agent import DQNFamily
from ddqn_agent.dqn_model import Net


class DQNAgent(DQNFamily):
    def __init__(self, conf, env):
        super().__init__(config=conf, env=env, model_type=Net)