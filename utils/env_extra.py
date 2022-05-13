import pandas as pd


def env_summary(env, agent=None, random=False):
    if random:
        state = env.random_state()
    else:
        state = env.reset()
    done = False
    state_info = env.get_state_info()
    state_infos = {key: [value] for key, value in state_info.items()}
    while not done:
        if agent is None:
            action = env.rand_action()
        else:
            action = agent.best_action(state)

        state, reward, done = env.step(action)
        state_info = env.get_state_info()

        for key, value in state_info.items():
            state_infos[key].append(value)
    return pd.DataFrame.from_dict(state_infos)
