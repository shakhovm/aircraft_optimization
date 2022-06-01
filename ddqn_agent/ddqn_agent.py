from dqn_family.general_agent import DQNFamily
from ddqn_agent.ddqn_model import LinearDDQNModel


class DDQNAgent(DQNFamily):
    def __init__(self, conf, env):
        super().__init__(config=conf, env=env, model_type=LinearDDQNModel)

