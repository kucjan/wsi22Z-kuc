import numpy as np
from random import random, choice


class Agent:
    def __init__(self, alpha, gamma, eps, env):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.env = env
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

    def update_qtable(self, curr_state, action, step_reward, next_state):
        curr_q = self.qtable[curr_state, action]
        max_next_q = np.max(self.qtable[next_state])

        self.qtable[curr_state, action] = curr_q + self.alpha * (
            step_reward * self.gamma * max_next_q - curr_state
        )

    def explore(self, curr_state):
        if random() < self.eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[curr_state])
