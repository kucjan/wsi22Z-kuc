import numpy as np


class QLearningSolver:
    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.curr_state = None

    def train(self):
        self.curr_state, _ = self.env.reset()
        done = False
        while not done:
            action = self.agent.explore(self.curr_state)
            next_state, step_reward, terminated, truncated, _ = self.env.step(action)
            self.agent.update_qtable(self.curr_state, action, step_reward, next_state)
            self.curr_state = next_state
            done = terminated or truncated

    def solve(self, random_action=False):
        self.curr_state, _ = self.env.reset()
        done = False
        steps = 0
        score = 0
        illegal_steps = 0
        while not done:
            action = self.agent.choose_action(random_action)
            next_state, step_reward, terminated, truncated, _ = self.env.step(action)
            self.curr_state = next_state
            steps += 1
            score += step_reward
            if step_reward == -10:
                illegal_steps += 1
            done = terminated or truncated
        return steps, score, illegal_steps

    def evaluate(self, max_iters, n_evals, n_episodes):
        self.curr_state, _ = self.env.reset()

        avg_scores = []
        avg_steps = []
        avg_illegal_steps = []
        eval_points = []

        learn_period = round(max_iters / n_evals)

        print("-" * 10 + " Evaluation " + "-" * 10)
        print(f"Learning period iterations: {learn_period}")
        print(f"Evaluation episodes: {n_episodes}")

        for period in range(1, max_iters + 1, learn_period):
            print(
                "-" * 10
                + f" Evaluation stop #{period % learn_period} at iteration #{period} "
                + "-" * 10
            )
            for _ in range(learn_period):
                self.train()
            print("TRENING SKONCZONY")
            period_steps = 0
            period_scores = 0
            period_illegal_steps = 0
            for episode in range(n_episodes):
                steps, score, illegal_steps = self.solve()
                period_steps += steps
                period_scores += score
                period_illegal_steps += illegal_steps
                print(
                    f"Episode #{episode}: steps: {steps}, score: {score}, illegal_steps: {illegal_steps}"
                )
            avg_steps.append(period_steps / n_episodes)
            avg_scores.append(period_scores / n_episodes)
            avg_illegal_steps.append(period_illegal_steps / n_episodes)
            eval_points.append(period)
        return eval_points, avg_steps, avg_scores, avg_illegal_steps
