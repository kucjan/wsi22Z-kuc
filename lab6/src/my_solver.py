import numpy as np


class QLearningSolver:
    def __init__(self, agent, illegal_step_reward):
        self.agent = agent
        self.illegal_step_reward = illegal_step_reward

    def train(self):
        curr_state, _ = self.agent.env.reset()
        done = False
        while not done:
            action = self.agent.explore(curr_state)
            next_state, step_reward, terminated, truncated, _ = self.agent.env.step(
                action
            )
            self.agent.update_qtable(curr_state, action, step_reward, next_state)
            curr_state = next_state
            done = terminated or truncated

    def solve(self, max_iters, random_action=False):
        curr_state, _ = self.agent.env.reset()
        done = False
        steps = 0
        score = 0
        illegal_steps = 0
        is_solved = True
        i = 0
        while not done:
            if i > max_iters:
                is_solved = False
                break
            action = self.agent.choose_action(curr_state, random_action)
            next_state, step_reward, terminated, truncated, _ = self.agent.env.step(
                action
            )
            curr_state = next_state
            steps += 1
            score += step_reward
            if step_reward == self.illegal_step_reward:
                illegal_steps += 1
            done = terminated or truncated
            i += 1
        return steps, score, illegal_steps, is_solved

    def evaluate(
        self,
        n_iters,
        n_evals,
        n_episodes,
        max_solve_iters,
        random_action=False,
    ):

        eval_values = np.zeros((n_evals, 4))
        eval_points = []

        learn_period = round(n_iters / n_evals)

        print("-" * 10 + " Evaluation " + "-" * 10)
        print(f"Learning period iterations: {learn_period}")
        print(f"Number of evalutaions: {n_evals}")
        print(f"Evaluation episodes: {n_episodes}\n")

        for period in range(1, n_iters + 1, learn_period):
            print(
                "-" * 10
                + f" Evaluation stop #{(period // learn_period) + 1} at iteration #{period} "
                + "-" * 10
            )
            for _ in range(learn_period):
                self.train()
            period_steps = 0
            period_scores = 0
            period_illegal_steps = 0
            period_solved = 0
            for episode in range(1, n_episodes + 1):
                steps, score, illegal_steps, is_solved = self.solve(
                    max_solve_iters, random_action
                )
                period_steps += steps
                period_scores += score
                period_illegal_steps += illegal_steps
                period_solved += is_solved
                print(
                    f"Episode #{episode}: steps: {steps}, score: {score}, illegal_steps: {illegal_steps}, solved: {is_solved}"
                )
            period_values = (
                np.array(
                    [
                        period_steps,
                        period_scores,
                        period_illegal_steps,
                        period_solved,
                    ]
                )
                / n_episodes
            )
            eval_values[period // learn_period] = period_values
            eval_points.append(period)
            print(
                "\nPeriod evaluation stats: avg_steps: {}, avg_scores: {}, avg_illegal_steps: {}, solved_ratio: {}\n".format(
                    period_values[0],
                    period_values[1],
                    period_values[2],
                    period_values[3],
                )
            )
        return (
            eval_points,  # points of evaluation
            eval_values[:, 0],  # avg_steps
            eval_values[:, 1],  # avg_scores
            eval_values[:, 2],  # avg_illegal_steps
            eval_values[:, 3],  # solved_ratios
        )
