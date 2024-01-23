import numpy as np

from utils.bdf_utils import bdf_second, f, set_weights


def bdf_so_solver(t, W, current_state, steps=2, step_size=1):
    """
    BDF Second-order Solver
    t: Timestep to evolve from
    W: Network weights
    current_state: Current state
    steps: No. of steps to evolve network -> (t, t+steps)
    step_size: step size
    """
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