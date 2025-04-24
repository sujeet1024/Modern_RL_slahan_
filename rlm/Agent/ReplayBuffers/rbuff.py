
import numpy as np
from numpy.random import choice



class RBuff:
    def __init__(self, capacity=5e3):
        self.buffer = [[]]
        self.max_buff_size = capacity
        self.tuple_size = None
        self.remove_olds = 1000
        # pass

    def add(self, datum:tuple):
        ''' datum: tuple of variable required for update computation, e.g. (s,a,r,s',terminal) for q learning'''
        if self.tuple_size==None:
            self.tuple_size = len(datum)
            self.buffer = [[] for i in range(self.tuple_size)]
        buffer_size = self.buffer_size()
        if buffer_size==self.max_buff_size:
            self.remove(min(self.remove_olds, buffer_size))
        for i in range(len(datum)):
            self.buffer[i].append(datum[i])

    def sample(self, sample_size:int=512):
        out_arr, buffer_size = [], self.buffer_size()
        sample_indices = np.random.choice(buffer_size, min(sample_size, buffer_size))
        for i in range(self.tuple_size):
            out_arr.append(np.array(self.buffer[i])[sample_indices])
        return out_arr

    def remove(self, samples_size=1):
        ''' FIFO, randomly remove oldest of samples '''
        buffer_size = self.buffer_size()
        remove_idx = np.sort(np.random.choice(min(buffer_size-1, 10*samples_size), min(buffer_size-1, samples_size)))[::-1]
        for i in range(self.tuple_size):
            for j in remove_idx:
                # breakpoint()
                self.buffer[i].pop(int(j))

    def buffer_size(self):
        return len(self.buffer[0])