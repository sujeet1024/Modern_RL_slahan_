import copy
import numpy as np
import pygame as pg
import gymnasium as gym
from gymnasium import spaces
from itertools import product

from rlm.Env import Environment, TransitionModel




class CliffWalking(Environment, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    def __init__(self, cols:int=12, rows:int=4, render_mode:str='human'):
        super().__init__()
        pg.init()
        self.cols = cols
        self.rows = rows
        self.render_mode = render_mode
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([cols-1, rows-1]), [2,])
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.steps = 0
        self.downhills = 0
        self.max_ep_len = 1e3
        self.transition = {0: (0,-1), 1: (1,0), 2: (0,1), 3: (-1,0)}
        self.action_anns = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
        self.cell_size = None
        self.window = None
        self.clock = None
        self.fps = self.metadata['render_fps']
        self.font = pg.font.SysFont("Georgia", 16)
        self.title_bar_height = 50
        self.grid = self.render_grid()
        self.agent_sym = self.agent_symbol()
        self.model = CliffStocModel(self.transition, self.cols, self.rows, 0.00)

    def start(self, start_loc:tuple=None):
        if start_loc is None:
            start_loc = (np.random.choice(self.cols-2), np.random.choice(self.rows-2))
        self.state = np.array(start_loc)
        return self.state

    def step(self, action):
        obs = np.clip(self.state + np.array(self.transition[action]), [0,0], [self.cols-1, self.rows-1], dtype=np.float32)
        self.state = obs
        pos_col, pos_row = obs
        r = -1
        done = False
        truncated = False
        if pos_row == self.rows-1:
            if 0<pos_col<self.cols-1:
                # done = True
                self.downhills += 1
                r = -100
                obs, _ = self.reset()
            elif pos_col == self.cols-1:
                done = True
                self.downhills = 0
        self.steps += 1
        if self.steps > self.max_ep_len:
            truncated = True
        self.render()
        info = {'step_number': self.steps, 'cliff_fallings': self.downhills}
        return obs, r, done, truncated, info

    def reset(self, **kwargs):
        self.state = np.array((0, 3), dtype=np.float32)
        self.steps = 0
        self.render()
        info = {'step_number': self.steps, 'cliff_fallings': self.downhills}
        return self.state, info
    
    def render(self,):
        grid = copy.copy(self.grid)
        grid.blit(self.agent_sym, (int(self.state[0])*self.cell_size, int(self.state[1])*self.cell_size+self.title_bar_height))
        display_info = self.font.render(f"Steps: {self.steps}", 0, (255,255,255))
        grid.blit(display_info, (8, min((self.title_bar_height-50)//2, 8)))
        array = pg.surfarray.array3d(grid)
        if self.render_mode == 'rgb_array':
            return np.transpose(array, (1,0,2))
        elif self.render_mode == 'human':
            # if configs.show_policy
            # return np.transpose(array, (1,0,2))
            if self.window is None:
                pg.display.init()
                self.clock = pg.time.Clock()
                self.window = pg.display.set_mode((array.shape[0], array.shape[1]))
            self.window.blit(grid, (0,0))
            pg.display.flip()
            self.clock.tick(self.fps)
            return None

    def agent_symbol(self):
        agent_sym = pg.Surface((self.cell_size,self.cell_size), pg.SRCALPHA)
        pg.draw.rect(agent_sym, (20, 230, 28),
                        (self.cell_size//6, 7*self.cell_size//24, 2*self.cell_size//3, 5*self.cell_size//12), width=2,
                        border_radius=7)
        pg.draw.rect(agent_sym, (20, 230, 28),
                        (self.cell_size//4, 9*self.cell_size//24, self.cell_size//2, 2*self.cell_size//12), width=2,
                        border_radius=3)
        pg.draw.line(agent_sym, (20, 230, 28),
                        (10*self.cell_size//27, 15*self.cell_size//24), (11*self.cell_size//27, 14*self.cell_size//24), width=1
                        )
        pg.draw.line(agent_sym, (20, 230, 28),
                        (13*self.cell_size//27, 15*self.cell_size//24), (14*self.cell_size//27, 14*self.cell_size//24), width=1
                        )
        pg.draw.line(agent_sym, (20, 230, 28),
                        (16*self.cell_size//27, 15*self.cell_size//24), (17*self.cell_size//27, 14*self.cell_size//24), width=1
                        )
        pg.draw.arc(agent_sym, (20, 230, 28),
                        (3*self.cell_size//8, self.cell_size//5, self.cell_size//4, self.cell_size//5),
                        0*3.14, 1*3.14, 2)
        return agent_sym
    
    def render_grid(self,):
        height, width = 400, 700
        rows, cols = 4, 12
        surface = pg.Surface((width, height))
        surface.fill((0,0,0))
        cell_size = min((height-50)/rows, width/cols)
        self.cell_size = cell_size
        grid_width, grid_height = cols*cell_size, rows*cell_size
        self.title_bar_height = height - grid_height
        for row in range(rows+1):
            pg.draw.line(surface, (255, 255, 255), (0, self.title_bar_height+row*cell_size), (grid_width, self.title_bar_height+row*cell_size))
        for col in range(cols+1):
            pg.draw.line(surface, (255, 255, 255), (col*cell_size, self.title_bar_height), (col*cell_size, self.title_bar_height+grid_height))
        colored_cells = {(rows-1, i): (128, 128, 128) for i in range(1, cols-1)}
        colored_cells[(rows-1, 0)] = (10, 10, 255),
        colored_cells[(rows-1, cols-1)] = (0, 255, 0)
        for (r, c), color in colored_cells.items():
            rect = pg.Rect(c*cell_size+3, self.title_bar_height+r*cell_size+3, cell_size-3, cell_size-3)
            pg.draw.rect(surface, color, rect)
        return surface
    
    def vis_policy(self, policy):
        grid = copy.copy(self.grid)
        display_info = self.font.render(f"Steps: {policy.iterations}", 0, (255,255,255))
        grid.blit(display_info, (8, min((self.title_bar_height-50)//2, 8)))
        
        action_vis = {'left': '←', 'right': '→', 'up': '↑', 'down': '↓'}
        obs_indices = (self.cols, self.rows)
        state_space = list(product(*(range(n) for n in obs_indices)))
        for state in state_space:
            action = policy.step(state)
            display_sa = self.font.render(f"{action_vis[self.action_anns[action]]}", 0, (255,255,255))
            grid.blit(display_sa, (int(state[0])*self.cell_size+self.cell_size/3, int(state[1])*self.cell_size+self.cell_size/3+self.title_bar_height))
        
        array = pg.surfarray.array3d(grid)
        if self.window is None:
            pg.display.init()
            self.clock = pg.time.Clock()
            self.window = pg.display.set_mode((array.shape[0], array.shape[1]))
        self.window.blit(grid, (0,0))
        pg.display.flip()
        self.clock.tick(self.fps)
    

class CliffStocModel(TransitionModel):
    def __init__(self, transitions, cols, rows, stoc_psa=0.01, **kwargs):
        super().__init__()
        self.transitions = transitions
        self.cols = cols
        self.rows = rows
        self.stoc_psa = np.clip(stoc_psa, 0, 1/len(self.transitions.keys()))

    def transition(self, obs, action, **kwargs):
        ''' returns list[rewards], list[next_obs], list[probs] '''
        pos_col, pos_row = obs
        actions = np.array(list(self.transitions.keys()))
        probs = np.ones(len(actions))*self.stoc_psa
        probs[action] += 1 - self.stoc_psa * len(actions)
        r = np.ones_like(probs)*(-1)
        nxt_obss = []
        if (pos_row==self.rows-1) and (pos_col==self.cols-1):
            return [0 for i in probs], [np.array(obs) for i in probs], (np.ones_like(probs)/len(probs)).tolist()
        for act in actions:
            # breakpoint()
            nxt_obss.append(np.clip(np.array(obs) + np.array(self.transitions[act]), [0,0], [self.cols-1, self.rows-1], dtype=np.float32))
            pos_col, pos_row = nxt_obss[act]
            if pos_row == self.rows-1:
                if 0<pos_col<self.cols-1:
                    r[act] = -100
                elif pos_col == self.cols-1:
                    r[act] = 0
        return r.tolist(), nxt_obss, probs.tolist()