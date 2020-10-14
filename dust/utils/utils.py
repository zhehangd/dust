import time

class Timer(object):
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent
        
    def __enter__(self):
        self.t = time.time()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t = time.time() - self.t
        if self.name in self.parent:
            self.parent[self.name] += self.t
        else:
            self.parent[self.name] = self.t
