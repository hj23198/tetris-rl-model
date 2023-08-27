import torch
import torch.nn as nn
from Env import Env
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
    
mem = Memory(100)
env = Env()
observation = env.reset()
for _ in range(100):
    action = torch.Tensor([random.randint(0, 8)])
    new_observation, reward, terminated, _, _ = env.step(action)
    mem.push(observation, action, new_observation, reward)
    observation = new_observation

    if terminated:
        env.reset()

sample = mem.sample(10)
sample = Transition(*zip(*sample))

state_batch = [[sample.state[x][y] for x in range(len(sample.state))] for y in range(4)]
state_batch = (torch.cat(state_batch[0]), torch.cat(state_batch[1]), torch.cat(state_batch[2]), torch.cat(state_batch[3]))
action_batch = torch.cat(sample.action)
reward_batch = torch.cat(sample.reward)
t=1
