import pickle
import torch
import random
from collections import deque
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory:
    def __init__(self, capacity) -> None:
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
with open('memory.pkl', 'rb') as f:
    mem = pickle.load(f)

memory = mem.memory
new_memory = deque(maxlen=100000)
for moment in memory:
    if moment[1] is not None:
    new_memory_item = Transition(moment[0], torch.Tensor(moment[1]), moment[2], moment[3])
    new_memory.append(new_memory_item)

mem.memory = new_memory_item
with open('memory.pkl', 'wb') as f:
    pickle.dump(mem, f)