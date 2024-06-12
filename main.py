import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pipe

import cv2
import time

import serial
import glob
import os
from pynput.keyboard import Listener
from memory_profiler import profile

from Network import Network
from utils.camera.run_cam import run_cam
from Gyro import run_gyro
from AbstractState import AbstractState
from PatternLearner import PatternLearner
from utils.check_gyro_balance import check_gyro_balance
from utils.bdf_utils import bdf_second, f, set_weights
from line_profiler import LineProfiler


# pattern training params (STDP)
walk_spike_times = [[800,900,1000,1100,1200,1300,
            1400,1500,1600],[300,400,500,600,700,800,900,1000],
            [0,100,200,300,400,500,600,700],
            [1000,1100,1200,1300,1400,1500,1600,1700,1800]]

curr_conn = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]

crawl_spike_times = [[0,100,200,300,400,500,600,700],[0,100,
            200,300,400,500,600,700],[800,900,1000,1100,1200,
            1300,1400,1500,1600],[800,900,1000,1100,1200,1300,
            1400,1500,1600]]

idle_weights = [[0, 0, 0, 0], [0, 0, 0, 0],
                [0, 0, 0, 0], [0, 0, 0, 0]]

# TMAX = 1000000
TMAX = 3500
N_neurons = 4

state_command = ''
state_name = ''
spikes_per_burst = 3

log_freq = 500

# save data
angular_velo_x_file = os.path.join('./data', 'angular_velo_x.txt')
angular_velo_y_file = os.path.join('./data', 'angular_velo_y.txt')
angular_velo_z_file = os.path.join('./data', 'angular_velo_z.txt')

crawl_times = []

camera_event_file = os.path.join('./data', 'cam_event.txt')


def on_press(key):
    try:
        k = key.char
    except:
        k = key.name

    global state_command, state_name
    if k in ['x', 'l', 'i', 's', 'd', 'e', 'g']: # [terminate, learn, idle, start]
        # global state_command
        print("State Command: ", k)
        state_command = k
        # global state_name
        # state_name = ''

    elif k in ['w', 'c']:
        # global state_name
        state_name = k

    else:
        state_command = ''
        state_name = ''


# def main(ser, gyro_pipe, cam_pipe, bdf_pipe):
def main(ser, cam_pipe, bdf_pipe):

    # keyboard event listener
    listener = Listener(on_press=on_press)
    listener.start()
    
    # Gyro inits
    # x_init, y_init, z_init = gyro_pipe.recv()
    x_init, y_init, z_init = 0, 0, 0
    x_list = []
    y_list = []
    z_list = []

    # control flags
    on_balance = True
    no_obstacle = True
    walk_pattern_id = 1
    crawl_pattern_id = 2
    new_pattern_id = 3

    idle_state = AbstractState(pattern_id=0, weights=idle_weights)
    learn_state = AbstractState(pattern_id=-1)
    walk_state = AbstractState(pattern_id=walk_pattern_id,
                        weights=[], V_state=[])
    crawl_state = AbstractState(pattern_id=crawl_pattern_id,
                        weights=[], V_state=[])
    

    current_state = idle_state
    current_state_id = idle_state.get_pattern_id()

    available_states = [idle_state.get_pattern_id(), 
        learn_state.get_pattern_id(), walk_state.get_pattern_id(),
        crawl_state.get_pattern_id()]
    
    # walk_state = None
    # crawl_state = None

    pattern_learner = PatternLearner()

    # network = Network(tmax=TMAX)

    actual_spike_ctr = [0, 0, 0, 0]

    steps = 2   # number of steps to compute evolve for
    steps_spiked_neurons = np.zeros((steps, N_neurons))   # spiked_neurons (steps, N_neurons)

    autonomous = False

    keep_crawling_init_t = 0
    keep_crawling = False
    crawl_dur = 8000
    # crawl_dur = 1000

    start_walk_transition_t = 0
    keep_walking = False
    walk_transition_dur = 4000
    first_walk = True

    gait_changed = False

    cam_event_times = []

    print("Initializing EnigmaXPetoi...")
    # for t in tqdm(range(steps, TMAX)):
    for t in range(steps, TMAX):
        global state_command
        # if t == 10:
        #     state_command = 'l'
        #     state_name = 'w'
        # if t == 15:
        #     state_command = 'l'
        #     state_name = 'c'
        if t == 20:
            state_command = 's'  
        # elif t == 500: # artificial
        # elif t in crawl_times: # artificial
        #     current_state = crawl_state
        #     current_state_id = crawl_state.get_pattern_id()

        # User input command
        # global state_command
        if state_command == 'x':
            print('Terminated')
            state_command = ''
            break
        elif state_command == 'l':
            current_state = learn_state
            current_state_id = learn_state.get_pattern_id()
            state_command = ''
        elif state_command == 'i':
            autonomous = False
            current_state = idle_state
            current_state_id = idle_state.get_pattern_id()
            state_command = ''
        elif state_command == 's':
            autonomous = True
            print('Starting autonomous navigation!')
            current_state = idle_state
            current_state_id = idle_state.get_pattern_id()
            state_command = ''
        elif state_command == 'd':
            print(f'Obstacle detected, I think its time for crawl at time {t}!')
            current_state = crawl_state
            current_state_id = crawl_state.get_pattern_id()
            state_command = ''
            crawl_times.append(t)
        elif state_command == 'e':
            print(f'I think its time for walk at time {t}!')
            current_state = walk_state
            current_state_id = walk_state.get_pattern_id()
            state_command = ''
            crawl_times.append(t)
            bdf_pipe.send("Walk")
        elif state_command == 'g':
            cam_pipe.send('save_img')
            state_command = ''
            

        # DSM
        # LEARNING
        if current_state_id == learn_state.get_pattern_id() and not autonomous:
            print("State: LEARN")
            # gyro_pipe.send("Learning")
            cam_pipe.send("Learning")

            if state_name == 'w' and not walk_state:
                print("LEARN WALK")
                walk_weights, V_state = pattern_learner.learn(walk_spike_times,
                        curr_conn, ser, walk_pattern_id, use_bdf=True)
                print('Walk Weights ', walk_weights, V_state)
                walk_state = AbstractState(pattern_id=walk_pattern_id,
                        weights=walk_weights, V_state=V_state)
                
                available_states.append(walk_pattern_id)
                # gyro_pipe.send("Finished learning")
                cam_pipe.send("Finished learning")
                # once converged transition to IDLE
                current_state_id = idle_state.get_pattern_id()
                current_state = idle_state

            elif state_name == 'c' and not crawl_state:
                print("LEARN CRAWL")
                crawl_weights, V_state = pattern_learner.learn(crawl_spike_times,
                        curr_conn, ser, crawl_pattern_id, use_bdf=True)

                # print('Crawl Weights ', crawl_weights, V_state)
                crawl_state = AbstractState(pattern_id=crawl_pattern_id,
                        weights=crawl_weights, V_state=V_state)
                
                available_states.append(crawl_pattern_id)

                # gyro_pipe.send("Finished learning")
                cam_pipe.send("Finished learning")
                # once converged transition to IDLE
                current_state_id = idle_state.get_pattern_id()
                current_state = idle_state

            else:   #TODO: Improve
                # print("NEW STATE")
                # new_spike_times, curr_conn = np.loadtxt(pattern_file)
                # new_weights, V_state = pattern_learner.learn(new_spike_times,
                #         curr_conn, ser, new_pattern_id, use_bdf=True)
                # new_state = AbstractState(pattern_id=new_pattern_id,
                #         weights=new_weights, V_state=V_state)
                # available_states.append(new_pattern_id)
                # new_pattern_id += 1
                pass


        # AUTONOMOUS PHASE
        # elif autonomous and walk_pattern_id in available_states and crawl_pattern_id in available_states:
        #     # start CPG evolution   
        elif autonomous:
            bdf_pipe.send("Start")

            if ser.in_waiting > 0:
                # Read the incoming bytes
                data = ser.read(2)  # Read 2 bytes
                # Combine the bytes to form the original integer
                number = (data[0] << 8) | data[1]
                print(f"Received integer: {number}")

            # if t == 30:
            #     cam_pipe.send("Init")
            # elif t == 500:
            #     cam_pipe.send("Mid")

            # one_hot_encoded = np.zeros(5)   # [neu1, neu2, neu3, neu4, motorCmdFlag]
            # spiked_neurons = np.zeros(4)
            
            # if t % steps == 0 :  # evolve network every t steps as we are computing for t steps
                #print('From main: ' ,current_state.get_pattern_id())
                # steps_spiked_neurons = network.evolve(t, current_state, use_bdf=True, steps=steps)
            
            # spiked_neurons = steps_spiked_neurons[t % steps]    # 2%2 = 0, 3%2 = 1
            
            # print("Neurons spiked main= ", spiked_neurons)

            # move once burst finishes
            # for cur_neu in range(N_neurons):
            #     if spiked_neurons[cur_neu] == 1:
            #         actual_spike_ctr[cur_neu] += 1

            #     if actual_spike_ctr[cur_neu] >= spikes_per_burst:
            #         actual_spike_ctr[cur_neu] = 0
            #         one_hot_encoded[cur_neu] = 1
            
            # poll gyro
            # if gyro_pipe.poll():
            #     x, y, z = gyro_pipe.recv()
            #     x -= x_init
            #     y -= y_init
            #     z -= z_init
            #     x_list.append(x)
            #     y_list.append(y)
            #     z_list.append(z)


            # if len(x_list) >= 5:
            #     # if not check_gyro_balance(x_list, y_list, z_list):
            #     if not check_gyro_balance(x_list[-5:], y_list[-5:], z_list[-5:]):
            #         on_balance = False
            #     else:
            #         on_balance = True

                # write data
                # if len(x_list) == 5:
                #     with open(angular_velo_x_file, "w") as file:
                #         for x_elem in x_list:
                #             file.write(str(x_elem) + '\n')

                #     with open(angular_velo_y_file, "w") as file:
                #         for y_elem in y_list:
                #             file.write(str(y_elem) + '\n')

                #     with open(angular_velo_z_file, "w") as file:
                #         for z_elem in z_list:
                #             file.write(str(z_elem) + '\n')
                # else:
                #     with open(angular_velo_x_file, "a") as file:
                #         file.write(str(x_list[-1]) + '\n')

                #     with open(angular_velo_y_file, "a") as file:
                #         file.write(str(y_list[-1]) + '\n')

                #     with open(angular_velo_z_file, "a") as file:
                #         file.write(str(z_list[-1]) + '\n')
                        
            
            
            #poll camera 
            if cam_pipe.poll():
                no_obstacle = not cam_pipe.recv() # obstacle present
                # cam_event_times.append(t)
                # if len(cam_event_times) == 1:
                #     with open(camera_event_file, "w") as file:
                #         file.write(str(cam_event_times[-1]) + '\n')
                # else:
                #     with open(camera_event_file, "a") as file:
                #         file.write(str(cam_event_times[-1]) + '\n')
            else:
                no_obstacle = True

            # WALK
            if current_state_id == walk_state.get_pattern_id():
                if t % log_freq == 0:
                    print(f"State: WALK, Timestep: {t}") 
                    # gyro_pipe.send("Finished learning")
            # if current_state_id == idle_state.get_pattern_id():
            # if current_state_id == 10:
                # print("State: WALK at timestep ", t)
                if not gait_changed or t in crawl_times:
                    gait_changed = True
                    bdf_pipe.send("Walk")
                    # cam_pipe.send("Walk")
                    print("Gait change to Walk!")

                # allows for transition time to walk
                if first_walk:
                    keep_walking = False
                elif not first_walk and t > (start_walk_transition_t +
                        walk_transition_dur):
                    keep_walking = False
                else:
                    keep_walking = True


                # if not on_balance or not no_obstacle and \
                #     not keep_walking:
                #     current_state_id = idle_state.get_pattern_id()
                #     idle_state = idle_state
                #     first_walk = False
                #     print(f"Obstacle detected at {t}")
                #     with open('data/obs_detect.txt', 'w') as file:
                #         file.write(str(t) + '\n')

            # CRAWL
            elif current_state_id == crawl_state.get_pattern_id():
            # elif current_state_id == idle_state.get_pattern_id():
                if t % log_freq == 0:
                    print(f"State: CRAWL, Timestep: {t}")
                if not gait_changed or t in crawl_times:
                # if not gait_changed:
                    keep_crawling_init_t = t
                    keep_crawling = True
                    gait_changed = True
                    bdf_pipe.send("Crawl")
                    # cam_pipe.send("Crawl")
                    print("Gait change to Crawl!")
                # network.set_weights(crawl_state.get_weights())
                if (on_balance and not no_obstacle) or keep_crawling:
                    if not keep_crawling: #First time obstacle detection
                        keep_crawling_init_t = t
                        keep_crawling = True
                    if t > keep_crawling_init_t + crawl_dur: #when crawl_dur runs out
                        keep_crawling = False
                    # if not np.all(one_hot_encoded[:4] == 0):
                    #     #print("LOL = ", one_hot_encoded)
                    #     one_hot_encoded[4] = 1
                    #     data_str = ','.join(map(str, one_hot_encoded.astype(int))) + '\n'
                    #     ser.write(data_str.encode())
                    #     time.sleep(0.1)
                # else:

                if (not on_balance or no_obstacle) and not keep_crawling:
                    current_state_id = idle_state.get_pattern_id()
                    current_state = idle_state

            # IDLE
            if current_state_id == idle_state.get_pattern_id():
                # if t % log_freq == 0:
                print(f"State: IDLE, Timestep: {t}")
                cam_pipe.send("Idle")
                # network.set_weights(idle_state.get_weights())

                # time.sleep(10)

                if on_balance and no_obstacle and \
                    walk_pattern_id in available_states:
                        # transition to WALK
                        gait_changed = False
                        current_state_id = walk_state.get_pattern_id()
                        current_state = walk_state

                        if not first_walk:
                            start_walk_transition_t = t
                            keep_walking = True
                            pass

                    
                elif on_balance and not no_obstacle and \
                    crawl_pattern_id in available_states:
                        # transition to CRAWL
                        gait_changed = False
                        current_state_id = crawl_state.get_pattern_id()
                        current_state = crawl_state

                        keep_crawling_init_t = t
                        keep_crawling = True


        else:
            # time.sleep(0.2)
            print("State: IDLE")

    cam_pipe.send('done')



# @profile
def run():
    parent_conn_cam, child_conn_cam = Pipe()
    parent_conn_bdf, child_conn_bdf = Pipe()

    ser = serial.Serial('/dev/ttyUSB0', 9600)
    ser.reset_input_buffer()
    
    # p1 = Process(target=main, args=(ser, parent_conn_gyro, parent_conn_cam, parent_conn_bdf))
    p1 = Process(target=main, args=(ser, parent_conn_cam, parent_conn_bdf))
    # p2 = Process(target=run_gyro, args=(child_conn_gyro,))
    p3 = Process(target=run_cam, args=(child_conn_cam,))
    p4 = Process(target=bdf_second, args=(ser, child_conn_bdf,))

    p1.start()
    # p2.start()
    p3.start()
    p4.start()
    
    p1.join()
    # p2.join()
    p3.join()
    p4.join()



if __name__ == '__main__':
    # parent_conn_gyro, child_conn_gyro = Pipe()
    # parent_conn_cam, child_conn_cam = Pipe()
    # parent_conn_bdf, child_conn_bdf = Pipe()

    # ser = serial.Serial('/dev/ttyUSB0', 9600)
    # ser.reset_input_buffer()
    
    # start = time.time()
    # # p1 = Process(target=main, args=(ser, parent_conn_gyro, parent_conn_cam, parent_conn_bdf))
    # p1 = Process(target=main, args=(ser, parent_conn_cam, parent_conn_bdf))
    # # p2 = Process(target=run_gyro, args=(child_conn_gyro,))
    # p3 = Process(target=run_cam, args=(child_conn_cam,))
    # p4 = Process(target=bdf_second, args=(ser, child_conn_bdf,))

    # p1.start()
    # # p2.start()
    # p3.start()
    # p4.start()
    
    # p1.join()
    # # p2.join()
    # p3.join()
    # p4.join()

    # print("Total Execution time = ", time.time()-start)

    # print("All processes are done.")

    start = time.time()
    # profiler = LineProfiler()
    # profiler.add_function(run)
    run()
    # profiler.run('run()')
    # profiler.print_stats()
    print("Total Execution time = ", time.time()-start)
