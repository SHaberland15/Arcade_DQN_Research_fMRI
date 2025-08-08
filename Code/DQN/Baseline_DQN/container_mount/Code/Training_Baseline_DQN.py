# Train the network
#start: python3 Training_Baseline_DQN.py -g breakout [, space_invaders, enduro]

import argparse
import os
from ale_python_interface import ALEInterface
import numpy as np
from PIL import Image
import torch
from torch import nn
import copy
from skimage.transform import resize


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


def ale_init(rng, game):
    
    max_frames_per_episode = 50000
    
    ale = ALEInterface()
    ale.loadROM(str.encode('/workspace/ROMs/' + game + '.bin'))
    ale.setInt(b'max_num_frames_per_episode', max_frames_per_episode)
    minimal_actions = ale.getMinimalActionSet()
    (screen_width,screen_height) = ale.getScreenDims()
    ale_seed = rng.integers(2^32)
    ale.setInt(b'random_seed', ale_seed)
    random_seed = ale.getInt(b'random_seed')
    ale.setFloat(b'repeat_action_probability', 0.0)
    action_repeat_prob = ale.getFloat(b'repeat_action_probability')
    
    print('width/height: ' +str(screen_width) + '/' + str(screen_height))
    print('max frames per episode: ' + str(max_frames_per_episode))
    print('minimal actions: ' + str(minimal_actions))
    print('random seed: ' + str(random_seed))
    print('action repeat prob.: ' + str(action_repeat_prob))
    
    return ale


def ale_15hz(ale, action):
    
    (screen_width,screen_height) = ale.getScreenDims()
    
    screen_vec_1 = np.empty((screen_height, screen_width), dtype=np.uint8)
    screen_vec_2 = np.empty((screen_height, screen_width), dtype=np.uint8)
    screen_max = np.zeros((screen_height, screen_width, 2), dtype=np.uint8)
    
    reward_sum = 0
    episode_end_flag = False
    
    reward = ale.act(action)
    reward_sum += reward
    if (ale.game_over()):
        episode_end_flag = True
    
    reward = ale.act(action)
    reward_sum += reward
    if (ale.game_over()):
        episode_end_flag = True
    
    reward = ale.act(action)
    reward_sum += reward
    if (ale.game_over()):
        episode_end_flag = True
    ale.getScreenGrayscale(screen_vec_1)
    
    reward = ale.act(action)
    reward_sum += reward
    if (ale.game_over()):
        episode_end_flag = True
    ale.getScreenGrayscale(screen_vec_2)
    
    screen_max[:,:,0] = screen_vec_1
    screen_max[:,:,1] = screen_vec_2
    
    return screen_max, reward_sum, episode_end_flag

def preproc_screen(screen):

    screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
    transformed_image = resize(screen_Gray_max, output_shape=(84, 84), anti_aliasing=None, 
                               preserve_range=True)
    int_image = np.asarray(transformed_image, dtype=np.float32)/255.0
   
    return int_image


def run_episode(ale, dqn_agent, step_no, no_op_max_this_episode, train_flag):
    
    ale.reset_game()
    episode_reward = 0.0
    action = 0
    no_op_action_count = 0
    
    while not ale.game_over():
        
        screen_Gray_15hz, reward_15hz, episode_end_flag = ale_15hz(ale, action)
        screen_preproc_15hz = preproc_screen(screen_Gray_15hz)
        
        if no_op_action_count < no_op_max_this_episode:
            action = 0
        else:
            action = dqn_agent(screen_preproc_15hz, reward_15hz, train_flag, episode_end_flag)
        no_op_action_count += 1
        
        episode_reward += reward_15hz
        step_no += 1
    
    return episode_reward, step_no


def print_report(ale, dqn_agent, epoch_no, n_training_epochs, episode_reward, episode_no, train_flag):
    
    episode_frame_number = ale.getEpisodeFrameNumber()
    frame_number = ale.getFrameNumber()
    
    print(' ')
    if train_flag:
        print('TRAINING:')
    else:
        print('EVALUATION:')
    print('Epoch no. ' + str(epoch_no + 1) + ' of ' + str(n_training_epochs))
    print('Frame Number: ' + str(frame_number) + '; Episode Frame Number: ' + str(episode_frame_number))
    print('Episode ' + str(episode_no) + ' ended with score: ' + str(episode_reward))
    print('action count: ' + str(dqn_agent.action_count))
    print('eps-greedy training: ' + str("{:.6f}".format(dqn_agent.eps_param)))
    print('eps-greedy evaluation: ' + str(dqn_agent.eps_param_eval))


class DqnAgent():
    
    def __init__(self, rng, minimal_actions):
        
        self.rng = rng
        self.minimal_actions = minimal_actions
        self.eps_param = 1.0
        self.eps_param_eval = 0.05
        self.experience_replay_buffer_size = 400000
        self.experience_replay_buffer_circle_ind = 0
        self.experience_replay_buffer = []
        self.state = np.zeros((4, 84, 84), dtype=np.float32)
        self.action_ind = 0
        self.action_count = 0
        self.network = DqnNN(len(minimal_actions)).to('cuda')
        self.network_target_update_freq = 10000
        self.train_network_stepsize = 4
        self.batch_size = 32
        self.discount_gamma = torch.tensor(0.99)
        self.lr_value = 0.00025
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr_value)
        self.clip_rewards_flag = True
        self.network_target = copy.deepcopy(self.network)
    
    
    def __call__(self, screen, reward, train_flag, episode_end_flag):
        
        state_new = self.update_state(screen)
        
        if train_flag:
            self.update_experience_replay_buffer(state_new, reward)
        
        action, self.action_ind = self.act_eps_greedy(state_new, train_flag)

        if episode_end_flag == True:
            self.state = np.zeros((4, 84, 84), dtype=np.float32)
        else:
            self.state = state_new
        
        if (train_flag and self.action_count >= self.experience_replay_buffer_size and 
                self.action_count % self.train_network_stepsize == 0):
            self.train_network()
        
        return action
    
    def update_state(self, screen):
    
        state_new = np.concatenate((np.expand_dims(screen, axis=0), self.state[0:3, :,:]))
        
        return state_new
    
    def update_experience_replay_buffer(self, state_new, reward):
        
        if len(self.experience_replay_buffer) < self.experience_replay_buffer_size:
            self.experience_replay_buffer.append((self.state, state_new, self.action_ind, reward))
        else:
            self.experience_replay_buffer[self.experience_replay_buffer_circle_ind] = (
                        (self.state, state_new, self.action_ind, reward)
                        )
        self.experience_replay_buffer_circle_ind = (
                        (self.experience_replay_buffer_circle_ind + 1) % self.experience_replay_buffer_size
                        )
    
    def act_eps_greedy(self, state_new, train_flag):
        
        if train_flag:
            eps_param_value = self.eps_param
        else:
            eps_param_value = self.eps_param_eval
        
        if self.rng.random() < eps_param_value:
            action_index = self.rng.integers(len(self.minimal_actions)) 
        else:
            state_tensor = torch.from_numpy(state_new).cuda().unsqueeze(0)
            with torch.no_grad():
                net_out = self.network(state_tensor)
            action_index = torch.argmax(net_out).cpu().numpy()
        
        action = self.minimal_actions[action_index]
        
        if train_flag:
            self.eps_param = max(1.0*(1.0 - self.action_count/1e6) + 0.1*self.action_count/1e6, 0.1)
            self.action_count += 1
        
        return action, action_index
    
    def sample_batch(self):
        
        state_batch_np = np.empty((self.batch_size, 4, 84, 84), dtype=np.float32)
        state_next_batch_np = np.empty((self.batch_size, 4, 84, 84), dtype=np.float32)
        action_ind_batch_np = np.empty(self.batch_size, dtype=np.int64)
        reward_batch_np = np.empty(self.batch_size, dtype=np.float32)
        
        for sample_no in range(self.batch_size):
            
            random_sample_ind = self.rng.integers(self.experience_replay_buffer_size)
            (state, state_next, action_ind, reward) = self.experience_replay_buffer[random_sample_ind]
            state_batch_np[sample_no,:,:,:] = state
            state_next_batch_np[sample_no,:,:,:] = state_next
            action_ind_batch_np[sample_no] = action_ind
            reward_batch_np[sample_no] = reward
        
        return state_batch_np, state_next_batch_np, action_ind_batch_np, reward_batch_np
    
    def clip_reward(self, reward_vec):
        
        for i in range(len(reward_vec)):
            if reward_vec[i] > 0.0:
                reward_vec[i] = 1.0
            elif reward_vec[i] < 0.0:
                reward_vec[i] = -1.0
        
        return reward_vec
    
    def train_network(self):
        
        if int(self.action_count) % self.network_target_update_freq == 0:
            self.network_target = copy.deepcopy(self.network)
        
        state_batch_np, state_next_batch_np, action_ind_batch_np, reward_batch_np = self.sample_batch()
        
        if self.clip_rewards_flag:
            reward_batch_np = self.clip_reward(reward_batch_np)
        
        state_batch_tensor = torch.from_numpy(state_batch_np).cuda()
        state_next_batch_tensor = torch.from_numpy(state_next_batch_np).cuda()
        
        with torch.no_grad():
            net_out_state_next = self.network_target(state_next_batch_tensor)
            action_ind_max = torch.argmax(net_out_state_next, axis=1)
            q_next_max = net_out_state_next[np.arange(self.batch_size), action_ind_max]
            q_pred = torch.from_numpy(reward_batch_np).cuda() + self.discount_gamma * q_next_max
        
        net_out_state = self.network(state_batch_tensor)
        q_actual = net_out_state[np.arange(self.batch_size), action_ind_batch_np]
        
        loss = self.loss_fn(q_actual, q_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', help='game name', type=str, required=True)
    args = parser.parse_args()
    
    rng = np.random.default_rng()
    
    ale = ale_init(rng, args.game)
    
    minimal_actions = ale.getMinimalActionSet()
    
    dqn_agent = DqnAgent(rng, minimal_actions)
    
    n_training_epochs = 200 
    n_steps_per_training_epoch = 250000
    n_steps_per_eval_epoch = 125000
    no_op_max_during_eval = 30
    weight_storage_stepsize = 10
    
    training_episode_no = 0
    eval_episode_no = 0

    DQN_reward = {}
    
    eval_reward_list_over_epochs = []
    
    for epoch_no in range(n_training_epochs):
    
        training_step_no = 0
        
        while training_step_no < n_steps_per_training_epoch:
        
            training_episode_no += 1
            no_op_max_this_episode = rng.integers(no_op_max_during_eval)
            
            episode_reward, training_step_no = (
                run_episode(ale, dqn_agent, training_step_no, no_op_max_this_episode, True)
                )
            
            print_report(ale, dqn_agent, epoch_no, n_training_epochs, 
                         episode_reward, training_episode_no, True)
        
        
        eval_step_no = 0
        
        while eval_step_no < n_steps_per_eval_epoch:
            
            eval_episode_no += 1
            no_op_max_this_episode = rng.integers(no_op_max_during_eval)
            
            episode_reward, eval_step_no = (
                        run_episode(ale, dqn_agent, eval_step_no, no_op_max_this_episode, False)
                        )

            eval_reward_list.append(episode_reward)
            
            print_report(ale, dqn_agent, epoch_no, n_training_epochs, episode_reward, 
                         eval_episode_no, False)
        
        eval_epoch_mean = np.mean(eval_reward_list)
        eval_reward_list_over_epochs.append(eval_epoch_mean)
        
        print(' ')
        print('**************')
        print('Eval Epoch mean: ' + str("{:.1f}".format(eval_epoch_mean)))
        print('**************')
        
        print(' ')
        print('Evals over epochs:')
        
        if (epoch_no + 1) % weight_storage_stepsize == 0:
            torch.save(dqn_agent.network.state_dict(), '/workspace/container_mount/checkpoints/breakout_DNN_epoch_' + str(epoch_no + 1).zfill(3) + '.pt')

            DQN_reward[str(epoch_no+1).zfill(3)] = eval_epoch_mean
            
            with open('/workspace/container_mount/checkpoints/DQN_reward_breakout.txt','a') as file:
                file.write('%s : %s\n' % (str(epoch_no+1).zfill(3), eval_epoch_mean))
                file.write('\n')

        for i in range(len(eval_reward_list_over_epochs)):
            print(str("{:.1f}".format(eval_reward_list_over_epochs[i])))
        
        print(' ')


if __name__ == '__main__':
    main()
