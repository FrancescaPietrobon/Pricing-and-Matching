from UCB import UCB
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCB_Matching(UCB):
    def __init__(self, n_arms, n_rows, n_cols):
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        assert n_arms == n_cols * n_rows

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
        return row_ind, col_ind

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1) + reward)/self.t


if __name__ == '__main__':
    from Environment import Environment             # Simulator
    import matplotlib.pyplot as plt

    #p = np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
    # rows: promo codes; columns: customers - values: price_item2 * (1-discount) * conversion_rate_item2_given1_promo (average)
    #           cust1 (c1)   cust2 (c1)   cust3 (c1)   cust4 (c2)   cust5 (c3)  ...
    #   P0
    #   P0
    #   P0
    #   P0
    #   P1
    #   P1
    #   P2
    #   P3
    #   ...

    price_item2 = 50
    prob_buy_item21 = np.array([[0.20, 0.30, 0.35, 0.25],
                                [0.30, 0.35, 0.40, 0.30],
                                [0.50, 0.45, 0.55, 0.40],
                                [0.70, 0.65, 0.75, 0.69]])
    discount_p1 = 0.1
    discount_p2 = 0.2
    discount_p3 = 0.5
    n_promos = 100
    daily_customers = np.array([30, 20, 25, 25])
    p = np.zeros((n_promos, sum(daily_customers)))
    #p.fill(-1e6)
    for row_index in range(n_promos):
        for column_index in range(sum(daily_customers)):
            if row_index < 0.7 * n_promos:
                if column_index < daily_customers[0]:
                    p[row_index, column_index] = price_item2 * prob_buy_item21[0][0]
                elif column_index < daily_customers[1]:
                    p[row_index, column_index] = price_item2 * prob_buy_item21[0][1]
                elif column_index < daily_customers[2]:
                    p[row_index, column_index] = price_item2 * prob_buy_item21[0][2]
                else:
                    p[row_index, column_index] = price_item2 * prob_buy_item21[0][3]
            elif row_index < 0.9 * n_promos:
                if column_index < daily_customers[0]:
                    p[row_index, column_index] = price_item2 * (1-discount_p1) * prob_buy_item21[1][0]
                elif column_index < daily_customers[1]:
                    p[row_index, column_index] = price_item2 * (1-discount_p1) * prob_buy_item21[1][1]
                elif column_index < daily_customers[2]:
                    p[row_index, column_index] = price_item2 * (1-discount_p1) * prob_buy_item21[1][2]
                else:
                    p[row_index, column_index] = price_item2 * (1-discount_p1) * prob_buy_item21[1][3]
            elif row_index < 0.97 * n_promos:
                if column_index < daily_customers[0]:
                    p[row_index, column_index] = price_item2 * (1-discount_p2) * prob_buy_item21[2][0]
                elif column_index < daily_customers[1]:
                    p[row_index, column_index] = price_item2 * (1-discount_p2) * prob_buy_item21[2][1]
                elif column_index < daily_customers[2]:
                    p[row_index, column_index] = price_item2 * (1-discount_p2) * prob_buy_item21[2][2]
                else:
                    p[row_index, column_index] = price_item2 * (1-discount_p2) * prob_buy_item21[2][3]
            else:
                if column_index < daily_customers[0]:
                    p[row_index, column_index] = price_item2 * (1-discount_p3) * prob_buy_item21[3][0]
                elif column_index < daily_customers[1]:
                    p[row_index, column_index] = price_item2 * (1-discount_p3) * prob_buy_item21[3][1]
                elif column_index < daily_customers[2]:
                    p[row_index, column_index] = price_item2 * (1-discount_p3) * prob_buy_item21[3][2]
                else:
                    p[row_index, column_index] = price_item2 * (1-discount_p3) * prob_buy_item21[3][3]


    opt = linear_sum_assignment(-p)
    n_exp = 1
    T = 365             # Days in a year (time horizon)
    regret_ucb = np.zeros((n_exp, T))
    for e in range(n_exp):
        learner = UCB_Matching(p.size, *p.shape)
        rew_UCB = []
        opt_rew = []
        env = Environment(p.size, p)
        for t in range(T):
            pulled_arms = learner.pull_arm()
            rewards = env.round(pulled_arms)
            learner.update(pulled_arms, rewards)
            rew_UCB.append(rewards.sum())
            opt_rew.append(p[opt].sum())
        regret_ucb[e, :] = np.cumsum(opt_rew)-np.cumsum(rew_UCB)

    plt.figure(0)
    plt.plot(regret_ucb.mean(axis=0))
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.show()
