#Gameplay for Space Invaders and Breakout black and white
#python3 Gameplay_Breakout_SpaceInvaders.py -sp /workspace/Task/ramdisk/ -pc 1 -g space_invaders -sn 0 -rp /workspace/ROMs/
#Modified Python code using the ALE python interface written by Ben Goodrich: https://github.com/bbitmaster/ale_python_interface

import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import time
import fcntl
from PIL import Image, ImageDraw, ImageFont
import argparse
from skimage.transform import resize
import socket

parser = argparse.ArgumentParser()

parser.add_argument('-sp', '--save_path', help='Save path', type=str,
                    required=True)
parser.add_argument('-pc', '--pseudo_code', help='Pseudonymization proband', 
                    type=str, required=True)
parser.add_argument('-g', '--game', help='Game name', type=str,
                    required=True)
parser.add_argument('-sn', '--session_no', help='Session number', type=str,
                    required=True)
parser.add_argument('-rp', '--rom_path', help='Rom path', type=str,
                    required=True)
args = parser.parse_args()

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)

def preproc_screen(screen):
    
    transformed_image = resize(screen, output_shape=(pixel_screen, pixel_screen), 
                               anti_aliasing=None, preserve_range=True)
    int_image = np.asarray(transformed_image, dtype=np.float32)

    return np.ascontiguousarray(int_image, dtype=np.float32)


def preproc_score(screen_np_in):  

    transformed_image = resize(screen_np_in, output_shape=(pixel_score, pixel_score), 
                               anti_aliasing=None, preserve_range=True) 
    screen_y_float_rescaled_np = np.array(transformed_image, dtype=np.float32)

    return screen_y_float_rescaled_np


#Initialization of the ALE environment using the ALE Interface can be found here:
#https://github.com/bbitmaster/ale_python_interface 


(display_width,display_height) = (1024,768)

pygame.display.init()
pygame.font.init()
screen = pygame.display.set_mode((display_width,display_height), pygame.FULLSCREEN)
pygame.display.set_caption("Arcade Learning Environment Player Agent Display")

pixel_screen = 84
pixel_height_screen = 74
index_array_screen = pixel_screen - pixel_height_screen
game_height = 666

if args.game == "breakout":
    scale_factor_score = 1.2
    scale_factor_screen = 1.2
    pixel_score = 114
    pixel_height_score = 15
    breakout_flag = True
else:
    scale_factor_score = 1.7
    scale_factor_screen = 1.4
    pixel_score = 105
    pixel_height_score = 13
    breakout_flag = False

game_surface = pygame.Surface((pixel_screen,pixel_height_screen)) 
game_surface_score = pygame.Surface((pixel_score,pixel_height_score))  
pygame.mouse.set_visible(False)

pygame.display.flip()

clock = pygame.time.Clock()

episode = 0
total_reward = 0.0
total_total_reward = 0.0

n_frames = 18900

loop_count = 0
loop_count_intro = 0
loop_15hz_count = 0
mod_4_count = 0
a = 0
a_old = 0

screen_temp = np.zeros((210, 160, 3), dtype=np.uint8)
screen_temp_Gray = np.zeros((210, 160), dtype=np.uint8)
screen_15hz_Gray = np.zeros((210, 160, 2, int(n_frames/4)), dtype=np.uint8)
screen_15hz_RGB = np.zeros((210, 160, 3, 2, int(n_frames/4)), dtype=np.uint8)
responses_vec = np.zeros(n_frames, dtype=np.uint8)
reward_vec = np.zeros(n_frames, dtype=np.double)
episode_vec = np.zeros(n_frames, dtype=np.int32)

waiting_for_trigger_flag = True
sec_left = 5

while(waiting_for_trigger_flag):

    screen.fill((0,0,0))

    font = pygame.font.SysFont("Ubuntu Mono",32)
    text = font.render("Session-No.: " + args.session_no, 1, (255,255,255))
    screen.blit(text,(380,310))
    text = font.render("Spiel startet in: " + str(sec_left), 1, (255,255,255))
    screen.blit(text,(380,410))

    if (sec_left == 0):

        waiting_for_trigger_flag = False

    pygame.display.flip()

    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    pressed = pygame.key.get_pressed()
    if(pressed[pygame.K_LALT]) and (pressed[pygame.K_q]):
        exit = True

    if(exit):
        break

    clock.tick(1.)
    sec_left -= 1


screen_preproc_old  = np.zeros((pixel_screen,pixel_screen)) 

while(loop_count < n_frames):

    mod_4_count += 1

    keys = 0
    pressed = pygame.key.get_pressed()
    keys |= pressed[pygame.K_u]
    keys |= pressed[pygame.K_7] <<1
    keys |= pressed[pygame.K_8] <<2
    keys |= pressed[pygame.K_9] <<3
    keys |= pressed[pygame.K_6] <<4
    a = key_action_tform_table[keys]

    if breakout_flag:
        if (a_old != 0):
            a = 0
        a_old = a
        
    responses_vec[loop_count] = a
    reward = ale.act(a);
    total_reward += reward
    total_total_reward += reward

    reward_vec[loop_count] = total_reward
    episode_vec[loop_count] = episode

    screen.fill((0,0,0))

    numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.uint8)
    ale.getScreenRGB(screen_temp)
    ale.getScreenGrayscale(screen_temp_Gray)

    screen_preproc_current = preproc_screen(screen_temp_Gray)
    screen_preproc_max = np.maximum(screen_preproc_current, 
                                    screen_preproc_old) 

    screen_preproc_scaled = screen_preproc_max*scale_factor_screen 
    screen_reversed = np.reshape(np.dstack((screen_preproc_scaled[:,:],
                                            screen_preproc_scaled[:,:], 
                                            screen_preproc_scaled[:,:], 
                                            np.zeros((pixel_screen, pixel_screen), 
                                            dtype=np.uint8)))[index_array_screen:,:],
                                            pixel_height_screen*pixel_screen*4) 
    numpy_surface[:] = screen_reversed
    screen.blit(pygame.transform.scale(game_surface, 
                                      (display_width,game_height)),
                                      (0,display_height-game_height))
        
    score_preproc = preproc_score(screen_temp_Gray) 
    score_preproc_max = score_preproc*scale_factor_score 
    numpy_surface_score = np.frombuffer(game_surface_score.get_buffer(),
                                        dtype=np.uint8)
    score_reversed = np.reshape(np.dstack((score_preproc_max[:,:], 
                                           score_preproc_max[:,:], 
                                           score_preproc_max[:,:], 
                                           np.zeros((pixel_score,pixel_score), 
                                           dtype=np.uint8)))[:pixel_height_score,:], 
                                           pixel_height_score*pixel_score*4) 
    numpy_surface_score[:] = score_reversed
    
    screen.blit(pygame.transform.scale(game_surface_score, 
                                      (display_width,display_height-game_height)),
                                      (0,0))
    
    if(mod_4_count == 3):
        screen_15hz_Gray[:,:,0,loop_15hz_count] = screen_temp_Gray
        screen_15hz_RGB[:,:,:,0,loop_15hz_count] = screen_temp
    if(mod_4_count == 4):
        mod_4_count = 0        
        screen_15hz_Gray[:,:,1,loop_15hz_count] = screen_temp_Gray
        screen_15hz_RGB[:,:,:,1,loop_15hz_count] = screen_temp
        loop_15hz_count += 1

    screen_preproc_old  = screen_preproc_current 

    del numpy_surface
    del numpy_surface_score
   
    pygame.display.flip()

    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(pressed[pygame.K_LALT]) and (pressed[pygame.K_q]):
        exit = True

    if(exit):
        break

    clock.tick(45.)

    if(ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) 
                               + " Episode Frame Number: "
                               + str(episode_frame_number))
        print("Episode " + str(episode) 
                         + " ended with score: " 
                         + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1

    loop_count += 1


end_loop = True
loop_count_outro = 0

while end_loop:

    screen.fill((0,0,0))

    font = pygame.font.SysFont("Ubuntu Mono",40)
    text = font.render("Block " + args.session_no + " completed!", 1, 
                      (255,255,255))
    screen.blit(text,(380,310))
    text = font.render("Total score: " + str(total_total_reward), 1, 
                      (255,255,255))
    height = font.get_height()*1.2
    screen.blit(text,(380,310 + height))

    pygame.display.flip()

    exit=False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break

    pressed = pygame.key.get_pressed()

    if(pressed[pygame.K_LALT]) and (pressed[pygame.K_q]):
        end_loop = False

    if(exit):
        break
    clock.tick(45.)

screen.fill((0,0,0))

font = pygame.font.SysFont("Ubuntu Mono",35)
text = font.render("Done, writing files...", 1, (255,255,255))
screen.blit(text,(380,410))

pygame.display.flip()

pygame.event.pump()

clock.tick(60.)

np.save(args.save_path + args.pseudo_code + '_' + args.game 
                       + '_' + args.mode + '_' + args.session_no 
                       + '_screen_15hz_Gray', screen_15hz_Gray)
np.save(args.save_path + args.pseudo_code + '_' + args.game 
                       + '_' + args.mode + '_' + args.session_no 
                       + '_screen_15hz_RGB', screen_15hz_RGB)

np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.mode + '_' + args.session_no 
                          + '_responses_vec.csv', responses_vec, 
                          delimiter=",", fmt='%01.0u')
np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.mode + '_' + args.session_no 
                          + '_reward_vec.csv', reward_vec, 
                          delimiter=",", fmt='%01.1f')
np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.mode + '_' + args.session_no 
                          + '_episode_vec.csv', episode_vec, 
                          delimiter=",", fmt='%01.0u')

file = open(args.save_path + args.pseudo_code + '_' + args.game 
                           + '_' + args.mode +'_total_reward.txt','a')
file.write('\n')
np.savetxt(file, np.array([total_total_reward]), delimiter=",", fmt='%01.0u')
file.close()
