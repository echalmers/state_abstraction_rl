class UpdatablePriorityQueue(dict):

    def insert(self, item, priority):
        """insert item with priority. If item already exists, update it's priority"""
        self[item] = priority

    def pop(self, item=None):
        """pop item from queue. If item not specified, pop highest-priority item"""
        key_to_pop = item or max(self, key=self.get)
        super().pop(key_to_pop)
        return key_to_pop

    def is_empty(self):
        return len(self) == 0

if __name__ == '__main__':
    PQueue = UpdatablePriorityQueue()
    PQueue.insert(((1, 1), 2), 5)
    PQueue.insert(((1, 2), 1), 6)
    PQueue.insert(((3, 4), 0), 1)
    PQueue.insert(((4, 1), 3), 3)
    print(PQueue, "\n")

    print("Popping off values:")
    print("(STATE, ACTION, PRIORITY)")
    while not PQueue.is_empty():
        print(PQueue.pop())