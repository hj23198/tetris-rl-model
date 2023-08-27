import pygame
import time
import sys
import numpy as np
from collections import namedtuple, deque
import random
import pickle
from Env import Env
import torch

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

def display(env, window, step):
    obs, reward, term, trunc, _ = env.step(step)
    if term or trunc:
       env.reset()
    
    env.render()

    pygame.event.pump()
    pygame.display.update()
    return obs, reward

pygame.init()
mem = Memory(10000)
window = pygame.display.set_mode((200, 620))
window.fill((0, 0, 0))
env = Env("human")
state = env.reset()


MOVE_LEFT = pygame.K_s
MOVE_RIGHT = pygame.K_f
ROTATE_LEFT = pygame.K_LEFT
ROTATE_RIGHT = pygame.K_RIGHT
HARD_DROP = pygame.K_d
SOFT_DROP = pygame.K_SPACE
HOLD = pygame.K_a
RESET = pygame.K_r
DAS_TIME = 0.5

left_last_pressed = 0
right_last_pressed = 0
hold_used = False
t = time.time()
while True:
    for event in pygame.event.get():
        step = None
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == MOVE_LEFT:
                step = 1
                left_last_pressed = time.time()
            
            if event.key == MOVE_RIGHT:
                step = 3
                right_last_pressed = time.time()

            if event.key == ROTATE_LEFT:
                step = 6
            
            if event.key == ROTATE_RIGHT:
                step = 7

            if event.key == HARD_DROP:
                step = 4
                hold_used = False

            if event.key == HOLD and not hold_used:
                step = 8
                hold_used = True

            if event.key == RESET:
                state = env.reset()
                mem.push(state, step, new_state, reward)

            if event.key == pygame.K_p:
                with open('memory.pkl', 'wb') as f:
                    pickle.dump(mem, f)
                print("Saved memory")

            if event.key == pygame.K_l:
                with open('memory.pkl', 'rb') as f:
                    mem = pickle.load(f)
                print("Loaded memory")

            if event.key == pygame.K_q:
                print("Memory size: ", len(mem))

        held_keys = pygame.key.get_pressed()
        if held_keys[SOFT_DROP]:
                step = 5

        if DAS_TIME < time.time() - left_last_pressed:
            if held_keys[MOVE_LEFT]:
                step = 0

        if DAS_TIME < time.time() - right_last_pressed:
            if held_keys[MOVE_RIGHT]:
                step = 2

        if step is not None:
            new_state, reward = display(env, window, step)
            mem.push(state, torch.unsqueeze(torch.Tensor([step]), dim=0), new_state, reward)





        
