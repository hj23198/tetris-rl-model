import gymnasium as gym
from Env import Env
import torch
from torch import nn
from collections import deque
import random
from collections import namedtuple
import math
from itertools import count
import pickle

env = Env(render_mode="human")
obs = env.reset()


# while True:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated:
#         obs = env.reset()
#     env.render()


device = ("cuda" if torch.cuda.is_available()
          else "cpu")

print(f"Using {device} device")

class TetrisNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.state_proccessing = nn.Sequential(nn.Conv2d(1, 4, (3, 3), padding=1),
                                                nn.MaxPool2d((2, 2)),
                                                nn.Sigmoid(),
                                                nn.BatchNorm2d(4),
                                                nn.Conv2d(4, 8, (3, 3), padding=1),
                                                nn.MaxPool2d((2, 2)),
                                                nn.Sigmoid(),
                                                nn.BatchNorm2d(8),
                                                nn.Conv2d(8, 16, (3, 3), padding=1),
                                                nn.Sigmoid(),
                                                nn.Flatten()
                                                )
        #output (16, 7, 2) [224]

        self.final_output = nn.Sequential(nn.Linear(278, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 256),
                                          nn.Sigmoid(),
                                          nn.Linear(256, 128),
                                          nn.Sigmoid(),
                                          nn.Linear(128, 9))

    def forward(self, x):
        # (state, piece_type, hold_piece, piece_queue)
        # state: (1, 30, 10)
        # piece_type: (1, 7, 1)
        # hold_piece: (1, 8, 1)
        # piece_queue: (1, 8, 5)
        state, piece_type, hold_piece, piece_queue = x
        piece_data = torch.concatenate((piece_type, hold_piece, self.flatten(piece_queue)), dim=1) # [1, 55]
        proccessed_state = self.state_proccessing(state)
        combined_data = torch.concatenate((proccessed_state, piece_data), dim=1) # [1, 279]
        output = self.final_output(combined_data)
        return output

class Memory:
    def __init__(self, capacity) -> None:
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 64
GAMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = Env("human")
state = env.reset()

n_actions = 9

policy_net = TetrisNet().to("mps")
target_net = TetrisNet().to("mps")
# policy_net.load_state_dict(torch.load("model2.pth"))
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
# with open('memory.pkl', 'rb') as f:
#     memory = pickle.load(f)
#     print("Loaded memory")
memory = Memory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshhold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshhold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.Tensor([[env.action_space.sample()]]).to("mps")
    
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    dtype=torch.bool).to("mps")
    
    non_final_next_states = tuple(map(torch.cat, zip(*[s for s in batch.next_state if s is not None])))
    

    state_batch = tuple(map(torch.cat, zip(*batch.state)))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64))
    next_state_values = torch.zeros(BATCH_SIZE).to("mps")
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss

    
for i_episode in range(1000):
    state = env.reset()
    total_reward = 0
    loss = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        #conv to tensor
        done = terminated or truncated
        total_reward += reward
        if terminated:
            next_state = None
        else:
            next_state = observation

        memory.push(state, action, next_state, reward)
        state = next_state
        loss += optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t+1)
            env.render()
            print(f"Episode: {i_episode} | Average Reward: {total_reward/episode_durations[-1]} | Average Loss: {loss/episode_durations[-1]}")
            total_reward = 0
            break

torch.save(policy_net.state_dict(), "model2.pth")







