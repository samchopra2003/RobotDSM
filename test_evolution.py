from Network import Network
from AbstractState import AbstractState
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

TMAX = 10000

network = Network(tmax=TMAX)
steps = 2
walk_pattern_id = 1
walk_weights = [[ 0.,         -0.29861238, -0.29001042, -0.30592875],
                [-0.29941841,  0.         ,-0.30300882, -0.30460323],
                [-0.30577322, -0.30759699 , 0.         ,-0.29377304],
                [-0.2895197 , -0.31566515 ,-0.30321847 , 0.        ]] 
V_state_walk = (-1.9328633044556662, -1.9300663127099382, -2.0307014121474727, -1.8995934228897688, -1.7863218340988563, -1.785088401370714, -1.9391517387341104, -1.931700967759835, -3.10620082659903, -3.0357013757016476, -2.8188212437434688, -1.5005196148041313, 2.142259269281092, 2.2005189059667516, -0.17103409313773324, -1.5342846647969157)

walk_state = AbstractState(pattern_id=walk_pattern_id,
                        weights=walk_weights, V_state=V_state_walk)
walk_state.set_weights(walk_weights)
walk_state.set_V_state(V_state_walk)
print(walk_state.get_pattern_id(),walk_state.get_V_state(),walk_state.get_weights())

crawl_weights = [[ 0.,          0.34068513, -0.31088257, -0.32189059],
                [ 0.25962229  ,0.         ,-0.28921189 ,-0.29080276],
                [-0.30813584 ,-0.31394906  ,0.          ,0.26952577],
                [-0.31807214 ,-0.3187071   ,0.34013083  ,0.        ]] 
V_state_crawl = (-1.2007550793204442, -1.2303606722161244, -0.6147559681362567, -1.2622134792566786, 2.1712398013752887, 2.1920332554666406, -0.1739955990271707, -1.3535909991080948, -2.7045607416090287, -2.74859062946178, -2.9028329179985892, -1.9322284278424033, -2.75116025219089, -2.783376934852111, -2.9672563316217584, -1.896613638484465)
crawl_state = AbstractState(pattern_id=walk_pattern_id,
                        weights=crawl_weights, V_state=V_state_crawl)
#crawl_state.set_weights(crawl_weights)
#crawl_state.set_V_state(V_state_crawl)
print(crawl_state.get_pattern_id(),crawl_state.get_V_state(),crawl_state.get_weights())

network.set_weights(crawl_state.get_weights())

steps_spiked_neurons = np.zeros((steps, 4)) 
spiked_neurons = np.zeros(4)
vm_full = [[] for x in range(4)]
t_ = np.linspace(0,TMAX//steps,TMAX//steps)
for t in tqdm(range(TMAX)):
    if t%steps == 0:
        steps_spiked_neurons,vm = network.evolve(t, crawl_state, use_bdf=True, steps=steps)
        for x in range(4):
            vm_full[x].append(vm[x])
        #print(vm)
    


    spiked_neurons = steps_spiked_neurons[t % steps]

    #if np.any(spiked_neurons == 1):
    print(t,spiked_neurons)
plt.plot(t_ , vm_full[0])
plt.plot(t_ , vm_full[1])
plt.plot(t_ , vm_full[2])
plt.plot(t_ , vm_full[3])
plt.show()