import rstris
import gymnasium as gym
import numpy as np
import pygame
import torch


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None, size=5) -> None:
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Dict({
            "board":gym.spaces.Box(0, 1, (30, 10), dtype=np.bool_),
            "current_piece":gym.spaces.Box(0, 1, (7, 1), dtype=np.bool_),
            "hold_piece":gym.spaces.Box(0, 1, (8, 1), dtype=np.bool_),
            "piece_queue":gym.spaces.Box(0, 1, (8, 5), dtype=np.bool_)
        })

        self.action_space = gym.spaces.Discrete(9)
        self.window = None
        self.clock = None

    def reset(self):
        super().reset()
        self._rstrisEnv = rstris.Env()
        observation = self._rstrisEnv.get_state()
        observation = torch.unsqueeze(torch.unsqueeze(torch.Tensor(observation[0]), dim=0), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[1]), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[2]), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[3]), dim=0).to("mps")
        return observation
    
    def step(self, action):
        observation, reward, terminated = self._rstrisEnv.step(action)
        observation = observation = torch.unsqueeze(torch.unsqueeze(torch.Tensor(observation[0]), dim=0), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[1]), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[2]), dim=0).to("mps"), torch.unsqueeze(torch.Tensor(observation[3]), dim=0).to("mps")
        reward = torch.Tensor([reward])
        info = {
            "is_terminated":terminated,
            "actions_taken":action
        }
        return observation, reward.to("mps"), terminated, False, info

    def render(self):
        board =  self._rstrisEnv.get_attached_state()
        board =torch.Tensor(board)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((200, 620))
            self.window.fill((0, 0, 0))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            canvas = pygame.Surface((self.window.get_width(), self.window.get_height()))
            for y in range(board.shape[0]):
                for x in range(board.shape[1]):
                    if board[y, x]:
                        pygame.draw.rect(canvas, (255, 255, 255), (x *  20, 600 - y * 20, 20, 20))

            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
