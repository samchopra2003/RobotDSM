import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pipe

import serial
import glob
import os
from pynput.keyboard import Listener

from Network import Network
from Camera import Camera
from Gyro import run_gyro
from AbstractState import AbstractState
from PatternLearner import PatternLearner
from utils.check_gyro_balance import check_gyro_balance

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

TMAX = 100000
N_neurons = 4

state_command = ''
state_name = ''
spikes_per_burst = 3

def on_press(key):
    try:
        k = key.char
    except:
        k = key.name

    global state_command, state_name
    if k in ['x', 'l', 'i', 's']: # [terminate, learn, idle, start]
        # global state_command
        print("State Command: ", k)
        state_command = k
        # global state_name
        state_name = ''

    elif k in ['w', 'c']:
        # global state_name
        state_name = k

    else:
        state_command = ''


def main(pipe):
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    ser.reset_input_buffer()

    # keyboard event listener
    listener = Listener(on_press=on_press)
    listener.start()
    
    # Pipe to comm with gyro
    x_init, y_init, z_init = pipe.recv()
    x_list = []
    y_list = []
    z_list = []
    
    cam = Camera()

    idle_state = AbstractState(pattern_id=0, weights=idle_weights)
    learn_state = AbstractState(pattern_id=-1)

    current_state = idle_state
    current_state_id = idle_state.get_pattern_id()

    available_states = [idle_state.get_pattern_id(), 
        learn_state.get_pattern_id()]
    
    walk_state = None
    crawl_state = None

    pattern_learner = PatternLearner()

    network = Network(tmax=TMAX)

    # control flags
    on_balance = True
    no_obstacle = True
    walk_pattern_id = 1
    crawl_pattern_id = 2
    new_pattern_id = 3

    actual_spike_ctr = [0, 0, 0, 0]

    steps = 2   # number of steps to compute evolve for
    steps_spiked_neurons = []   # spiked_neurons (steps, N_neurons)

    autonomous = False

    print("Initializing EnigmaXPetoi...")
    for t in tqdm(range(steps, TMAX)):
        # User input command
        global state_command
        if state_command == 'x':
            print('Terminated')
            break
        elif state_command == 'l':
            current_state = learn_state
            current_state_id = learn_state.get_pattern_id()
        elif state_command == 'i':
            autonomous = False
            current_state = idle_state
            current_state_id = idle_state.get_pattern_id()
        elif state_command == 's':
            autonomous = True


        # DSM
        # LEARNING
        if current_state_id == learn_state.get_pattern_id() and not autonomous:
            print("State: LEARN")
            pipe.send("Learning")

            if state_name == 'w' and not walk_state:
                print("LEARN WALK")
                # walk_spike_times, curr_conn = np.loadtxt(pattern_file)
                walk_weights, V_state = pattern_learner.learn(walk_spike_times,
                        curr_conn, ser, walk_pattern_id, use_bdf=True)
                walk_weights = np.array([[0,-0.3,-0.3,-0.3],
                                [-0.3,0,-0.3,-0.3],
                                [-0.3,-0.3,0,-0.3],
                                [-0.3,-0.3,-0.3,0]])
                walk_state = AbstractState(pattern_id=walk_pattern_id,
                        weights=walk_weights, V_state=V_state)
                available_states.append(walk_pattern_id)

                pipe.send("Finished learning")
                # once converged transition to IDLE
                current_state_id = idle_state.get_pattern_id()
                current_state = idle_state

            elif state_name == 'c' and not crawl_state:
                print("LEARN CRAWL")
                # crawl_spike_times, curr_conn= np.loadtxt(pattern_file)
                crawl_weights, V_state = pattern_learner.learn(crawl_spike_times,
                        curr_conn, ser, crawl_pattern_id, use_bdf=True)
                crawl_weights = np.array([[0,0.3,-0.3,-0.3],
                                        [0.3,0,-0.3,-0.3],
                                        [-0.3,-0.3,0,0.3],
                                        [-0.3,-0.3,0.3,0]])
                crawl_state = AbstractState(pattern_id=crawl_pattern_id,
                        weights=crawl_weights, V_state=V_state)
                available_states.append(crawl_pattern_id)

                pipe.send("Finished learning")
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

            # pipe.send("Finished learning")
            # # once converged transition to IDLE
            # current_state_id = idle_state.get_pattern_id()
            # current_state = idle_state
        
        # IDLE
        elif current_state_id == idle_state.get_pattern_id():
            print("State: IDLE")
            network.set_weights(idle_state.get_weights())

            #TODO: Extend for autonomous navigation

        # AUTONOMOUS PHASE
        elif autonomous and walk_pattern_id in available_states and crawl_pattern_id in available_states:
            print('Starting autonomous navigation!')
            one_hot_encoded = np.zeros(5)   # [neu1, neu2, neu3, neu4, motorCmdFlag]
            spiked_neurons = np.zeros(4)

            if t % steps == 0:  # evolve network every t steps as we are computing for t steps
                steps_spiked_neurons = network.evolve(t, current_state, use_bdf=True, steps=steps)
                spiked_neurons = steps_spiked_neurons[t % steps]    # 2%2 = 0, 3%2 = 1

            print("Neurons spiked = ", spiked_neurons)

            # move once burst finishes
            for cur_neu in range(N_neurons):
                if spiked_neurons[cur_neu] == 1:
                    actual_spike_ctr[cur_neu] += 1

                if actual_spike_ctr[cur_neu] >= spikes_per_burst:
                    actual_spike_ctr[cur_neu] = 0
                    one_hot_encoded[cur_neu] = 1

            # WALK
            if walk_pattern_id in available_states:
                if current_state_id == walk_state.get_pattern_id():
                    print("State: WALK")
                    network.set_weights(walk_state.get_weights())
                    if on_balance and no_obstacle:
                        data_str = ','.join(map(str, one_hot_encoded.astype(int))) + '\n'
                        ser.write(data_str.encode())
                    else:
                        current_state_id = idle_state.get_pattern_id()
                        idle_state = idle_state

            # CRAWL
            elif crawl_pattern_id in available_states:
                if current_state_id == crawl_state.get_pattern_id():
                    print("State: CRAWL")
                    network.set_weights(crawl_state.get_weights())
                    if on_balance and no_obstacle:
                        one_hot_encoded[4] = 1
                        data_str = ','.join(map(str, one_hot_encoded.astype(int))) + '\n'
                        ser.write(data_str.encode())
                    else:
                        current_state_id = idle_state.get_pattern_id()
                        current_state = idle_state

            else:
                if on_balance and no_obstacle and \
                    walk_pattern_id in available_states:
                        # transition to WALK
                        current_state_id = walk_state.get_pattern_id()
                        current_state = walk_state
                
                elif on_balance and not no_obstacle and \
                    crawl_pattern_id in available_states:
                    # transition to CRAWL
                    current_state_id = crawl_state.get_pattern_id()
                    current_state = crawl_state
        
            if pipe.poll():
                x, y, z = pipe.recv()
                x -= x_init
                y -= y_init
                z -= z_init
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

        
            if len(x_list) >= 5:
                if not check_gyro_balance(x_list, y_list, z_list):
                    on_balance = False
                else:
                    on_balance = True
            # on_balance = True
            
            # poll camera if walk and jump states available (after learning)
            if cam.check_obstacle():
                no_obstacle = False
            else:
                no_obstacle = True
            # no_obstacle = True
    
    state_command = ''
        

        
if __name__ == '__main__':
    parent_conn, child_conn = Pipe()

    image = glob.glob('./img/*.png')
    for img in image:
        os.remove(img)
    
    p1 = Process(target=main, args=(parent_conn,))
    p2 = Process(target=run_gyro, args=(child_conn,))

    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

    print("All processes are done.")
