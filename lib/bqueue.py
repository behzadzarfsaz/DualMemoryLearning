class Bqueue(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.lst = []

    def push(self, st):
        if len(self.lst) == self.max_size:
            self.lst.pop(0)
        self.lst.append(st)

    def get_list(self):
        return self.lst

    def is_empty(self):
        return len(self.lst) == 0

    def is_full(self):
        return len(self.lst) == self.max_size
