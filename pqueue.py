class StateActionPQueue:
    """
    A priority queue of state, action pairs. 
    Follows the format (STATE, ACTION, PRIORITY) where STATE is a tuple (x, y).
    
    Inspired from https://www.geeksforgeeks.org/priority-queue-in-python/.

    Example usage:
        PQueue = StateActionPQueue()
        PQueue.insert((1,1), 2, 5)
        while not PQueue.is_empty():
            print(PQueue.pop())
    """

    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def __check_state_action_exists(self, state, action):
        return [x for x in self.queue if x[0] == state and x[1] == action]

    def is_empty(self):
        return len(self.queue) == 0

    # inserts a (state, action, priority) triple 
    def insert(self, state, action, priority):
        if not self.__check_state_action_exists(state=state, action=action):
            self.queue.append((state, action, priority))

    # finds the value with the max priority and removes it from the queue
    def pop(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.queue[i][2] > self.queue[max_val][2]:
                    max_val = i
            item = self.queue[max_val]
            del self.queue[max_val]
            return item
        except IndexError:
            exit()

if __name__ == '__main__':
    PQueue = StateActionPQueue()
    PQueue.insert((1, 1), 2, 5)
    PQueue.insert((1, 2), 1, 6)
    PQueue.insert((3, 4), 0, 1)
    PQueue.insert((4, 1), 3, 3)
    print(PQueue, "\n")

    print("Popping off values:")
    print("(STATE, ACTION, PRIORITY)")
    while not PQueue.is_empty():
        print(PQueue.pop())