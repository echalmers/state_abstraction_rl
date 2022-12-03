import tables
import pqueue
import numpy as np
import random
from matplotlib import pyplot as plt


class MBRL:
    def __init__(
        self,
        states,
        action_space,
        actions,
        epsilon,
        discount_factor,
        theta_threshold,
        max_pqueue_loops=0,
        q_default=100,
        r_default=0,
        c_default=0,
        display_graphs=True
    ):
        self.states = states
        self.action_space = action_space
        self.actions = actions
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.theta_threshold = theta_threshold
        self.q_default = q_default
        self.total_episode_reward = 0
        self.q_table_updates = 0
        self.display_graphs = display_graphs
        self.plt = plt

        # Create and initialize StateActionTables, TTable, and PQueue.
        self.Q = tables.StateActionTable(default_value=self.q_default)
        self.R = tables.StateActionTable(default_value=r_default)
        self.C = tables.StateActionTable(default_value=c_default)
        self.T = tables.TTable()
        self.PQueue = pqueue.UpdatablePriorityQueue()

        if display_graphs:
            self.q_fig, self.q_ax = self.plt.subplots(1, 2)
            self.plt.ion()
            self.heatmap = np.full((states[1], states[0]), fill_value=0, dtype=int)


    def reset_total_episode_reward(self):
        self.total_episode_reward = 0


    def choose_action(self, s, sampled_action):
        if random.uniform(0, 1) < self.epsilon:
            return sampled_action
        else:
            return self.Q.get_best_action(state=tuple(s), actions=self.actions)


    def update(self, s, a, s_prime, r):
        self.total_episode_reward += r

        # update T, C, and R tables
        self.T[tuple(s), a, tuple(s_prime)] += 1
        self.C[tuple(s), a] += 1
        self.R[tuple(s), a] += (r - self.R[tuple(s), a]) / self.C[tuple(s), a]

        priority = abs(r + self.discount_factor * \
                max(self.Q.get_action_values(state=tuple(s_prime), actions=self.actions).values()) - self.Q[tuple(s), a])

        if priority > self.theta_threshold:
            self.PQueue.insert((tuple(s), a), priority)

        self.__empty_priority_queue()

        if self.display_graphs:
            self.heatmap[s_prime[1], s_prime[0]] += 1
            self.__update_graphs()


    def __empty_priority_queue(self):
        while not self.PQueue.is_empty():
            (x1, y1), act = self.PQueue.pop()

            self.Q[(x1, y1), act] = self.R[(x1, y1), act]

            # Loop for all (state, action) pairs predicted to lead to S:
            for (x2, y2) in self.T.get_states_accessible_from((x1, y1)):

                self.Q[(x1, y1), act] += self.discount_factor * (self.T[(x1, y1), act, (x2, y2)] / self.C[(x1, y1), act]) * \
                            max(self.Q.get_action_values(state=(x2, y2), actions=self.actions).values())
                self.q_table_updates += 1

            for (xbar, ybar), act_from_sbar_to_s in self.T.get_state_actions_with_access_to((x1, y1)):
                predicted_reward = self.R[(xbar, ybar), act_from_sbar_to_s]

                # Set priority for (sbar, act) pair
                priority = abs(predicted_reward + self.discount_factor * \
                    max(self.Q.get_action_values(state=(x1, y1), actions=self.actions).values()) - self.Q[(xbar, ybar), act_from_sbar_to_s])

                if priority > self.theta_threshold:
                    self.PQueue.insert(((xbar, ybar), act_from_sbar_to_s), priority)


    def __update_graphs(self):
        self.q_ax[0].cla()
        self.q_ax[1].cla()
        value_map = (self.Q.convert_to_np_arr(self.states + (self.action_space,), self.q_default)).max(axis=2).transpose()
        self.q_ax[0].set_title('perceived state values (max Q values)')
        self.q_ax[1].set_title('agent heat map')
        self.q_ax[0].imshow(value_map)
        self.q_ax[1].imshow(self.heatmap, cmap='hot')
        self.plt.pause(0.00001)
