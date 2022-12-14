import tables
import pqueue
import random
import sys

"""
MBRL: 

A model-based reinforcement learner that accepts a variety of input parameters 
and customization. This learner uses prioritized sweeping to efficiency update
Q values after an action is done.

CUSTOMIZABLE PARAMETERS:
- states
- action_space
- actions
- epsilon
- discount_factor
- theta_threshold
- max_pqueue_loops (NOT IMPLEMENTED YET)
- q_default
- r_default
- c_default
- display_graphs

PUBLIC METHODS:
- reset_total_episode_reward()
- choose_action(state, sampled_action)
- update(state, action, s_prime, reward)

"PRIVATE" METHODS
- __update_graphs()
- __empty_priority_queue()
"""
class MBRL:
    """
    A tabular model-based reinforcement learner that uses prioritized sweeping to efficiency update
    Q values after each action.
    """
    def __init__(
        self,
        actions,
        epsilon=0.1,
        discount_factor=0.9,
        theta_threshold=0,
        max_value_iterations=sys.maxsize,
        q_default=100,
        r_default=0,
        c_default=0
    ):
        """
        Creates a model based reinforcement learner with the parameters specified
        :param actions: sequence of actions available
        :param epsilon: exploration factor
        :param discount_factor: discount factor
        :param theta_threshold: temporal differences greater than this will be added to the priority queue for updating
        :param max_value_iterations: max value calculations per step
        :param q_default: default q value
        :param r_default: default r value
        :param c_default: starting count for transition counts
        """
        self.actions = actions
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.theta_threshold = theta_threshold
        self.max_value_iterations = max_value_iterations
        self.q_default = q_default
        self.q_table_updates = 0

        # Create and initialize StateActionTables, TTable, and PQueue.
        self.Q = tables.StateActionTable(default_value=self.q_default)
        self.R = tables.StateActionTable(default_value=r_default)
        self.C = tables.StateActionTable(default_value=c_default)
        self.T = tables.TTable()
        self.PQueue = pqueue.UpdatablePriorityQueue()

    def choose_action(self, s):
        """ 
        Chooses an action from a specific state uses e-greedy exploration.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.Q.get_best_action(state=tuple(s), actions=self.actions)

    def update(self, s, a, s_prime, r):
        """
        Updates the Q, T, C, and R tables appropriately by conducting prioritized sweeping
        in order to only update the important states first.
        """
        # update T, C, and R tables
        self.T[tuple(s), a, tuple(s_prime)] += 1
        self.C[tuple(s), a] += 1
        self.R[tuple(s), a] += (r - self.R[tuple(s), a]) / self.C[tuple(s), a]

        priority = abs(r + self.discount_factor * \
                max(self.Q.get_action_values(state=tuple(s_prime), actions=self.actions).values()) - self.Q[tuple(s), a])

        if priority > self.theta_threshold:
            self.PQueue.insert((tuple(s), a), priority)

        self.__process_priority_queue()

    def __process_priority_queue(self):
        for _ in range(min(len(self.PQueue), self.max_value_iterations)):

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
