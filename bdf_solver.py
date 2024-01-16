import numpy as np

from utils.bdf_utils import bdf_second, f, set_weights

# init V
# vm1_0, vf1_0, vs1_0, vus1_0 = -1,0,0,0
# vm2_0, vf2_0, vs2_0, vus2_0 = -1,0,0,0
# vm3_0, vf3_0, vs3_0, vus3_0 = -1,0,0,0
# vm4_0, vf4_0, vs4_0, vus4_0 = -1,0,0,0
# V = (vm1_0, vf1_0, vs1_0, vus1_0,
#        vm2_0, vf2_0, vs2_0, vus2_0,
#        vm3_0, vf3_0, vs3_0, vus3_0,
#        vm4_0, vf4_0, vs4_0, vus4_0)

def bdf_so_solver(t, W, current_state, steps=2, step_size=1):
    """
    BDF Second-order Solver
    t: Timestep to evolve from
    W: Network weights
    current_state: Current state
    steps: No. of steps to evolve network -> (t, t+steps)
    step_size: step size
    """
    # global V
    V = current_state.get_V_state()
    set_weights(W)
    
    [out, t] = bdf_second(f, V, step_size, t, t + steps)
    V = tuple([out[i][-1] for i in range(len(out))])

    # (steps, N_neurons) -> 2x4
    all_outs = []
    for i in range(steps):
        cur_out = []
        for j in range(len(V) // 4):
            cur_out.append(out[j][i])
        all_outs.append(cur_out)

    current_state.set_V_state(V)

    return all_outs