from math import ceil
import numpy as np


class Solver:
    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.curr_state = None

    def solve(self, train=False):
        self.curr_state, info = self.env.reset()
        done = False
        steps = 0
        score = 0
        illegal_steps = 0
        while not done:
            if train:
                action = self.agent.explore(self.curr_state)
            else:
                action = np.argmax(self.agent.qtable[self.curr_state])
            next_state, step_reward, terminated, truncated, _ = self.env.step(action)
            self.agent.update_qtable(self.curr_state, action, step_reward, next_state)
            self.curr_state = next_state
            steps += 1
            score += step_reward
            if step_reward == -10:
                illegal_steps += 1
            done = terminated or truncated
        return steps, score, illegal_steps

    def evaluate(self, n_episodes, n_evaluations, max_iters):
        self.curr_state, info = self.env.reset()

        avg_scores = []
        avg_steps = []
        avg_illegal_steps = []
        ev_points = []

        learn_period = ceil(max_iters / n_episodes)

        for period in range(0, max_iters, learn_period):
            for i in range(learn_period):
                self.solve(train=True)
            period_steps = 0
            period_scores = 0
            period_illegal_steps = 0
            for episode in range(n_episodes):
                steps, score, illegal_steps = self.solve(train=False)
                period_steps += steps
                period_scores += score
                period_illegal_steps += illegal_steps
            avg_steps.append(period_steps / n_episodes)
            avg_scores.append(period_scores / n_episodes)
            avg_illegal_steps.append(period_illegal_steps / n_episodes)
            ev_points.append(period)
        return ev_points, avg_scores, avg_steps, avg_illegal_steps
