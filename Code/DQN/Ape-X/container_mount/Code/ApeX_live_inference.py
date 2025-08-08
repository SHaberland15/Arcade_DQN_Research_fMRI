#Ape-X DQN plays one of the Atari games live with video output
#Example usage: python3 ApeX_live_inference.py  -r /workspace/ROMs/breakout.bin

#This code contrains portions originating from neka-nat and is licensed under the MIT License. See: https://github.com/neka-nat/distributed_rl
#Code, Functions and structures were adopted from the original repository. Original code is marked with [*]. 

import argparse
import os
import numpy as np
from ale_python_interface import ALEInterface
import pygame
from itertools import count
from collections import deque
import torch
from distributed_rl.distributed_rl.libs import models, utils
import time
import fcntl
import copy
from collections import deque
import gym
from gym.envs.atari.atari_env import AtariEnv
from gym import spaces
import random
from PIL import Image
import math
from skimage.transform import resize


class RunALE:
    
    def __init__(self, rom_path, noop, eps_greedy, rng):
        
        self.noop = noop
        self.eps_greedy = eps_greedy
        self.rng = rng
        self.display_width = 800
        self.display_height = 600
        self.noop_frame_count_15hz = 0
        
        # init ALE
        self.ale = ALEInterface()
        max_frames_per_episode = self.ale.getInt(b'max_num_frames_per_episode')
        print('ALE max frames per episode: ' + str(max_frames_per_episode))
        
        random_int = self.rng.integers(2^32)
        self.ale.setInt(b'random_seed', random_int)
        ale_seed = self.ale.getInt(b'random_seed')
        print('ALE random seed: ' + str(ale_seed))
        
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        action_repeat_prob = self.ale.getFloat(b'repeat_action_probability')
        print('ALE action repeat prob.: ' + str(action_repeat_prob))
        
        self.ale.loadROM(str.encode(rom_path))
        legal_actions = self.ale.getMinimalActionSet()
        print('ALE legal actions:')
        print(legal_actions)
        
        (screen_width, screen_height) = self.ale.getScreenDims()
        self.screen_width = screen_width
        self.screen_height = screen_height
        print('ALE screen dims: ' + str(screen_width) + '/' + str(screen_height))
        print(' ')
        
        # init Pygame
        pygame.display.init()
        pygame.font.init()
        self.pygame_display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('ALE Display')
        self.game_surface = pygame.Surface((screen_width, screen_height))
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
                
        # init variables

        self.screen_vec_Gray = np.empty((4,screen_height, screen_width), dtype=np.uint8)
        self.screen_vec_RGB = np.empty((4,screen_height, screen_width,3), dtype=np.uint8)
    
    def ale_15hz(self, action):   
        reward_sum = 0
        episode_end_flag = False

        screen_Gray = np.zeros((210, 160,2), dtype=np.uint8)
        
        for i in range(4): 
            reward = self.ale.act(action)
            reward_sum += reward
            self.ale.getScreenRGB(self.screen_vec_RGB[i])
            self.ale.getScreenGrayscale(self.screen_vec_Gray[i])
            if(self.ale.game_over()):
                episode_end_flag = True
            screen_Max = self.takeMax(self.screen_vec_RGB[i-1],self.screen_vec_RGB[i])  
            self.pygame_step(screen_Max)

        screen_Gray[:,:,0] = self.screen_vec_Gray[2]
        screen_Gray[:,:,1] = self.screen_vec_Gray[3]
        
        if episode_end_flag:
            self.noop_frame_count_15hz = 0
            self.ale.reset_game()

        return screen_Gray, reward_sum, episode_end_flag

    def takeMax(self, PrevScreen, CurrentScreen):

        screen_max_R = np.amax(np.dstack((PrevScreen[:,:,0], 
                                          CurrentScreen[:,:,0])), axis=2)
        screen_max_G = np.amax(np.dstack((PrevScreen[:,:,1], 
                                          CurrentScreen[:,:,1])), axis=2)
        screen_max_B = np.amax(np.dstack((PrevScreen[:,:,2], 
                                          CurrentScreen[:,:,2])), axis=2)
        return np.dstack((screen_max_R, screen_max_G, screen_max_B))

    
    def pygame_step (self,screen_Max):
        
        pygame.event.pump()  
        self.pygame_display.fill((0,0,0))
        numpy_surface = np.frombuffer(self.game_surface.get_buffer(),dtype=np.uint8)

        screen_temp_reversed = np.reshape(np.dstack((screen_Max[:,:,2], 
                                                     screen_Max[:,:,1], 
                                                     screen_Max[:,:,0], 
                                                     np.zeros((self.screen_height, self.screen_width), 
                                                     dtype=np.uint8))), 210*160*4)
        numpy_surface[:] = screen_temp_reversed
        del numpy_surface
        self.pygame_display.blit(pygame.transform.scale(self.game_surface, 
                                                    (self.display_width, 
                                                    self.display_height)),(0,0))
        pygame.display.flip()
        self.clock.tick(100.)


    def __call__(self, action):
        
        self.noop_frame_count_15hz += 1
        
        u_greedy = self.rng.random()
        if u_greedy <= self.eps_greedy:
            action = self.rng.integers(18)
        
        if self.noop_frame_count_15hz*4 <= self.noop: 
            action = 0
        
        screen_Gray, reward_sum, episode_end_flag = self.ale_15hz(action)
        
        return screen_Gray, reward_sum, episode_end_flag
        
class RunApeX(object):
    
    def epsilon_greedy(self, state, policy_net, eps = 0.005): #[*]
        if random.random() > eps:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1).cpu()
        else:
            return torch.tensor([random.randrange(policy_net.n_action)], dtype=torch.long)

    def initialize(self):
        nx_st = self.env.reset()
        nx_st_gray = np.empty((210, 160), dtype=np.uint8)
        self.env.ale.getScreenGrayscale(nx_st_gray) 
        nx_screen = np.empty((210, 160, 2), dtype=np.uint8)
        nx_screen[:,:,0] = nx_st_gray
        nx_screen[:,:,1] = nx_st_gray
        for _ in range(self.img_buf.maxlen):
            self.img_buf.append(self.preproc_state(nx_screen))
        for _ in range(np.random.randint(1, self.no_op_steps)):
            self.env.step(0)

    def preproc_state(self, screen):

        screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
        transformed_image = resize(screen_Gray_max, output_shape=(84, 84), anti_aliasing=None, 
                                   preserve_range=True)
        int_image = np.asarray(transformed_image, dtype=np.float32)/255.0
       
        return np.ascontiguousarray(int_image, dtype=np.float32)
    
    def reset(self):
        self.img_buf.clear()
        self.initialize()
        return np.array(list(self.img_buf))


    def __init__(self, game_name):
                
        if game_name == 'enduro.bin':
            self.env = gym.make('EnduroNoFrameskip-v4') 
            self.NumberActions = self.env.action_space.n
            self.policy_net = models.DuelingDQN(self.NumberActions).to('cuda')
            self.policy_net.load_state_dict(torch.load('/workspace/container_mount/checkpoints/'
                                                       'enduro/model_80000.pth'))
            self.policy_net.eval()
        elif game_name == 'breakout.bin':
            self.env = gym.make('BreakoutNoFrameskip-v4') 
            self.NumberActions = self.env.action_space.n
            self.policy_net = models.DuelingDQN(self.NumberActions).to('cuda')
            self.policy_net.load_state_dict(torch.load('/workspace/container_mount/checkpoints/'
                                                       'breakout/model_120000.pth'))
            self.policy_net.eval()
        elif game_name == 'space_invaders.bin':
            self.env = gym.make('SpaceInvadersNoFrameskip-v4') 
            self.NumberActions = self.env.action_space.n
            self.policy_net = models.DuelingDQN(self.NumberActions).to('cuda')
            self.policy_net.load_state_dict(torch.load('/workspace/container_mount/checkpoints/'
                                                       'space_invaders/model_260000.pth'))
            self.policy_net.eval()
        else:
            raise FileExistsError(game_name + ' not found!')
        
        self.eps_param_eval = 0.005
        self.buf_size = 4
        self.img_buf = deque(maxlen=self.buf_size)
        self.no_op_steps = 3
        self.state = self.reset()
    
    def __call__(self, screen_Gray, episode_end_flag):
        last_15hz_screen_max_84x84 = self.preproc_state(screen_Gray)

        self.img_buf.append(last_15hz_screen_max_84x84)
        self.state = np.array(list(self.img_buf))
        action = self.epsilon_greedy(torch.from_numpy(self.state).unsqueeze(0).to('cuda'), 
                                          self.policy_net, self.eps_param_eval)
        action_scalar = action.item() 
        
        return action_scalar

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ROM_path', help='ROM file path', type=str, required=True)

    args = parser.parse_args()
    
    game_name = os.path.basename(args.ROM_path)
    
    rng = np.random.default_rng()
    n_frames_60hz = 108000
    n_frames_15hz = n_frames_60hz//4
    noop = rng.integers(30)
    eps_greedy = 0.001
    action = 0
    
    episode_no = 1
    episode_reward = 0.0
    session_reward = 0.0
    
    episode_no_vec_15hz = np.zeros(n_frames_15hz, dtype=np.int32)
    episode_reward_vec_15hz = np.zeros(n_frames_15hz, dtype=np.double)
    
    run_ale = RunALE(args.ROM_path, noop, eps_greedy, rng)
    run_ape_x = RunApeX(game_name)
    
    for loop_count_15hz in range(n_frames_15hz):
    
        screen_Gray, reward, episode_end_flag = run_ale(action)
        
        episode_reward += reward
        session_reward += reward
        episode_reward_vec_15hz[loop_count_15hz] = episode_reward
        episode_no_vec_15hz[loop_count_15hz] = episode_no
        
        action = run_ape_x(screen_Gray, episode_end_flag)
        action = run_ale.ale.getMinimalActionSet()[int(action)] 

        if (loop_count_15hz + 1) % 250 == 0:
            print('Total Frame Number: ' + str((loop_count_15hz + 1)*4))
            print('Episode Number: ' + str(episode_no))
            print('Episode Score: ' + str(episode_reward))
            print(' ')
        
        if episode_end_flag:
            print('EPISODE END!')
            print('Total Frame Number: ' + str((loop_count_15hz + 1)*4))
            print('Episode Number: ' + str(episode_no))
            print('Episode Score: ' + str(episode_reward))
            print(' ')
            episode_reward = 0
            episode_no += 1
    
    print('Total session score: ' + str(session_reward))
    print(' ')

if __name__ == '__main__':
    main()

