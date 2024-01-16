import numpy as np

from bdf_solver import bdf_so_solver


class Network:
    def __init__(self, W=[[0, 0, 0, 0], [0, 0, 0, 0],
        [0, 0, 0, 0], [0, 0, 0, 0]], N_neurons=4,
        alpha=[-2, 2, -1.5, 2], delta=[0, 0, -0.88, 0], 
        beta=[-2, 2, -1.5, 2], Cm = 1, Iapp=[-2, -2, -2, -2], 
        tauf=1, taus=50, tauus=50*50, R=1,
        dT=1, Erev=0.48, Vthresh=2.5, scale=1, tmax=100000):
        self.W = W
        self.N_neurons = N_neurons
        self.Iapp = Iapp

        self.alpha = alpha
        self.delta = delta
        self.beta = beta

        self.R = 1
        self.Cm = Cm

        self.Vm = np.zeros((N_neurons,tmax))
        self.Vf = np.zeros((N_neurons,tmax))
        self.Vs = np.zeros((N_neurons,tmax))
        self.Vus = np.zeros((N_neurons,tmax))

        self.tauf = tauf
        self.taus = taus
        self.tauus = tauus
        self.dT = dT

        self.Erev = Erev
        self.Vthresh = Vthresh
        self.scale = scale

    
    def set_weights(self, W):
        self.W = W

    def get_weights(self):
        return self.W.copy()
    
    def sf(self, x, b, ds):
        """ Sigmoid synapse function. """
        k=b*(x-ds)
        return 1/(1+np.exp(-k))


    def evolve(self, t, current_state, use_bdf=True, steps=2):
        """ Evolve the membrane voltage of neurons. """
        spiked = [[0, 0, 0, 0] for _ in range(steps)]   # (steps, N_neurons)
        if use_bdf:
            for idx, vm in enumerate(bdf_so_solver(t, self.W, current_state, steps=steps)):
                for cur_neu in range(4):
                    if vm[cur_neu] > self.Vthresh:
                        spiked[idx][cur_neu] = 1

        else:
            spiked = [0, 0, 0, 0]
            for cur_neu in range(self.N_neurons):
                dVfdT = (self.Vm[cur_neu][t - 1] - self.Vf[cur_neu][t - 1]) / self.tauf
                dVsdT = (self.Vm[cur_neu][t - 1] - self.Vs[cur_neu][t - 1]) / self.taus
                dVusdT = (self.Vm[cur_neu][t - 1] - self.Vus[cur_neu][t - 1]) / self.tauus

                self.Vf[cur_neu][t] = self.Vf[cur_neu][t - 1] + (dVfdT * self.dT)
                self.Vs[cur_neu][t] = self.Vs[cur_neu][t - 1] + (dVsdT * self.dT)
                self.Vus[cur_neu][t] = self.Vus[cur_neu][t - 1] + (dVusdT * self.dT)

                F_N = self.alpha[0] * np.tanh((self.Vf[cur_neu][t - 1] - self.delta[0]) * self.scale)
                S_P = self.alpha[1] * np.tanh((self.Vs[cur_neu][t - 1] - self.delta[1]) * self.scale)
                S_N = self.alpha[2] * np.tanh((self.Vs[cur_neu][t - 1] - self.delta[2]) * self.scale)
                US_N = self.alpha[3] * np.tanh((self.Vus[cur_neu][t - 1] - self.delta[3]) * self.scale)

                Isum = 0
                for conn in range(self.N_neurons):                
                    Isum += (self.W[cur_neu][conn] * self.sf(self.Vs[conn][t], 2, 1))
                
                I_x = F_N + S_P + S_N + US_N
                I_P = (self.Vm[cur_neu][t - 1] / self.R)


                dVmdT = (- I_P - I_x + Isum) / self.Cm
                self.Vm[cur_neu][t] = self.Vm[cur_neu][t - 1] + (dVmdT * self.dT)

                if self.Vm[cur_neu][t] > self.Vthresh and self.Vm[cur_neu][t-1] < self.Vthresh:
                    spiked[cur_neu] = 1

        return spiked