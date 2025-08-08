#The Baseline DQN plays one of the Atari games live with video output

#start with: python3 Baseline_DQN_live_inference.py -r /workspace/ROMs/enduro.bin
#start with: python3 Baseline_DQN_live_inference.py -r /workspace/ROMs/space_invaders.bin
#start with: python3 BAseline_DQN_live_inference.py -r /workspace/ROMs/breakout.bin

import argparse
import os
import numpy as np
from ale_python_interface import ALEInterface
import pygame
from itertools import count
from collections import deque
import torch
from torch import nn
import time
import fcntl
import copy
from collections import deque
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

    
    def pygame_step(self,screen_Max):
        
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
        self.clock.tick(60.)


    def __call__(self, action):
        
        self.noop_frame_count_15hz += 1
        
        u_greedy = self.rng.random()
        if u_greedy <= self.eps_greedy:
            action = self.rng.integers(18)
        
        if self.noop_frame_count_15hz*4 <= self.noop: 
            action = 0
        
        screen_Gray, reward_sum, episode_end_flag = self.ale_15hz(action)
        
        return screen_Gray, reward_sum, episode_end_flag

class DqnNN(nn.Module):
    
    def __init__(self, no_of_actions):
        super(DqnNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, no_of_actions))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out

        
class RunSimpleDQN(object):
 

    def __init__(self, rng, game_name, minimal_actions):

        self.rng = rng
        self.minimal_actions = minimal_actions
        self.eps_param_eval = 0.05
        self.state = np.zeros((4, 84, 84), dtype=np.float32)
        self.action_ind = 0
        self.action_count = 0
        self.network = DqnNN(len(minimal_actions)).to('cuda')
        self.discount_gamma = torch.tensor(0.99)
        self.network_target = copy.deepcopy(self.network)


    def __call__(self, screen, reward):

        last_15hz_screen_max_84x84 = self.preproc_state(screen)
        state_new = self.update_state(last_15hz_screen_max_84x84)
        action, self.action_ind = self.act_eps_greedy(state_new)
        self.state = state_new

        return action

    def preproc_state(self, screen):

        screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
        transformed_image = resize(screen_Gray_max, output_shape=(84, 84), anti_aliasing=None, 
                                   preserve_range=True) 
        int_image = np.asarray(transformed_image, dtype=np.float32)/255.0

        return int_image


    def update_state(self, screen):
    
        state_new = np.concatenate((np.expand_dims(screen, axis=0), self.state[0:3, :,:]))
        
        return state_new
    
    def act_eps_greedy(self, state_new):

        if self.rng.random() < self.eps_param_eval:
            action_index = self.rng.integers(len(self.minimal_actions)) 
        else:
            state_tensor = torch.from_numpy(state_new).cuda().unsqueeze(0)
            with torch.no_grad():
                net_out = self.network(state_tensor)
            action_index = torch.argmax(net_out).cpu().numpy()
        
        action = self.minimal_actions[action_index]
        
        return action, action_index


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ROM_path', help='ROM file path', type=str, required=True)
    args = parser.parse_args()

    game_name = os.path.basename(args.ROM_path)
    rng = np.random.default_rng()
    noop = rng.integers(30)
    eps_greedy = 0.001
    run_ale = RunALE(args.ROM_path, noop, eps_greedy, rng)
    minimal_actions = run_ale.ale.getMinimalActionSet()
    dqn_agent = RunSimpleDQN(rng, game_name, minimal_actions)

    if game_name == 'enduro.bin':
        dqn_agent.network.load_state_dict(torch.load('/workspace/container_mount/checkpoints/enduro/' 
                                                      'enduro_DNN_epoch_240.pt'))
        dqn_agent.network.eval()
    elif game_name == 'breakout.bin':
        dqn_agent.network.load_state_dict(torch.load('/workspace/container_mount/checkpoints/breakout/' 
                                                      'breakout_DNN_epoch_240.pt'))
        dqn_agent.network.eval()
    elif game_name == 'space_invaders.bin':
        dqn_agent.network.load_state_dict(torch.load('/workspace/container_mount/checkpoints/space_invaders/' 
                                                      'space_invaders_DNN_epoch_240.pt'))
        dqn_agent.network.eval()
    else:
        raise FileExistsError(game_name + ' not found!')

    n_frames_60hz = 108000
    n_frames_15hz = n_frames_60hz//4
    action = 0
    
    episode_no = 1
    episode_reward = 0.0
    session_reward = 0.0
    
    episode_no_vec_15hz = np.zeros(n_frames_15hz, dtype=np.int32)
    episode_reward_vec_15hz = np.zeros(n_frames_15hz, dtype=np.double)
        
    
    for loop_count_15hz in range(n_frames_15hz):
    
        screen_Gray, reward, episode_end_flag = run_ale(action)
        
        episode_reward += reward
        session_reward += reward
        episode_reward_vec_15hz[loop_count_15hz] = episode_reward
        episode_no_vec_15hz[loop_count_15hz] = episode_no
        
        action = dqn_agent(screen_Gray, episode_end_flag)

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

if __name__ == '__main__':
    main()
