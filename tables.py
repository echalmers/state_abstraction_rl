import itertools
import numpy as np

class StateActionTable:
    """
    A table of values that can be looked up by state and action.
    Example usage:
        Q = StateActionTable(default_value=10)
        Q['s1', 'a1'] = 5.2
        Q['s1', 'a2'] = 2
        print(Q['s2', 'a5'])  # prints 10
    """

    def __init__(self, default_value):
        self.default_value = default_value
        self.table = dict()

    def __getitem__(self, item):
        return self.table.get(item[0], dict()).get(item[1], self.default_value)

    def __setitem__(self, key, value):
        self.table.setdefault(key[0], dict())[key[1]] = value

    def get_action_values(self, state, actions):
        """
        get dict of action values for the specified state
        :param state: state to get action values for
        :param actions: list of actions of interest
        :return: dictionary of action -> value
        """
        return {action: self[state, action] for action in actions}

    def get_best_action(self, state, actions):
        action_values = self.get_action_values(state=state, actions=actions)
        return max(action_values, key=action_values.get)

    def convert_to_np_arr(self, size, fill_value, dtype):
        arr = np.full(size, fill_value=fill_value, dtype=dtype)
        for (x, y) in list(self.table.keys()):
            for act, act_value in enumerate(list(self.get_action_values((x, y), [0, 1, 2, 3]).values())):
                arr[x, y, act] = act_value
        return arr

class TTable:
    """
    A table of values that can be looked up by state, action, and next state
    can infer both successor and predecessor states for a given state
    Example usage:
        T = TTable()
        T['s1', 'a1', 's2'] += 1
        T['s1', 'a2', 's3'] += 1
        T['s1', 'a1', 's4'] += 1
        T.get_states_accessible_from('s1')  # returns ['s2', 's4', 's3']
        T.get_states_with_access_to('s3')  # returns ['s1']
    """

    def __init__(self):
        self.forward_map = dict()
        self.backward_map = dict()

    def __getitem__(self, item):
        return self.forward_map.get(item[0], dict()).get(item[1], dict()).get(item[2], 0)

    def __setitem__(self, key, value):
        self.forward_map.setdefault(key[0], dict()).setdefault(key[1], dict())[key[2]] = value
        self.backward_map.setdefault(key[2], dict()).setdefault(key[1], dict())[key[0]] = value

    def get_states_accessible_from(self, state):
        """
        returns states that can be accessed from the specified state (inferred from table entries)
        :param state: state to infer successor states for
        :return: list of states
        """
        return list(itertools.chain(*[x.keys() for x in self.forward_map[state].values()]))

    def get_states_with_access_to(self, state):
        """
        returns states that have access to the specified state (inferred from table entries)
        :param state: state to infer predecessor states for
        :return: list of states
        """
        return list(itertools.chain(*[x.keys() for x in self.backward_map[state].values()]))

if __name__ == '__main__':
    # use StateActionTable class for Q, R, C tables
    R = StateActionTable(default_value=10)
    R['s1', 'a1'] = 5.2
    R['s1', 'a2'] += 2
    print('R entry for s1, a1: ', R['s1', 'a1'])
    print('R entry for s2, a2: ', R['s1', 'a2'])
    print('R entries for s1, actions a1-a3: ', R.get_action_values(state='s1', actions=['a1', 'a2', 'a3']))
    print('Best action choice (max):', R.get_best_action(state='s1', actions=['a1', 'a2', 'a3']))

    # use TTable class for the T table
    T = TTable()
    T['s1', 'a1', 's2'] += 1
    T['s1', 'a2', 's3'] += 1
    T['s1', 'a1', 's4'] += 1
    print('states accessible from s1: ', T.get_states_accessible_from('s1'))
    print('states with access to s3: ', T.get_states_with_access_to('s3'))