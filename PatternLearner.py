import numpy as np
from tqdm import tqdm

from utils.bdf_learner import train_bdf_second


class PatternLearner:
    def __init__(self, learning_time=40000, R=1, 
                N_neurons=4, k=0):
        self.learning_time = learning_time
        self.R = R
        self.N_neurons = N_neurons

        # Synaptic weights list
        self.W = np.random.normal(loc=0.0, scale=0.3,
            size=(self.N_neurons, self.N_neurons))
        np.fill_diagonal(self.W, 0)

    
    def train_STDP(self, lrn_rate, tpre, tpost, tau_learn):
        return lrn_rate*np.exp(np.sign(tpre-tpost)*(tpost-tpre)/tau_learn)
    

    def exp_dec(self, scale_syn, t):
        return np.exp(-1**t)*scale_syn
    

    def sf(self, x, b, ds):
        k=b*(x-ds)
        return 1/(1+np.exp(-k))


    def learn(self, ideal_spike_times, pattern_curr_conn,
            ser, pattern_id, use_bdf=True):
        if use_bdf:
            # conv_weights, V_state = train_bdf_second(pattern_id, ser)
            # train_state.set_weights(conv_weights)
            # train_state.set_V_state(V_state)
            return train_bdf_second(pattern_id, ser)

        else:
            # spike times (2-D list) for STDP
            # shape: (number of neurons, spikes/burst)
            alpha = [-2, 2, -1.5, 2]
            delta = [0, 0, -0.88, 0]
            Cm =  1
            Iapp=[-2, -2, -2, -2]
            I = np.zeros((self.N_neurons,self.learning_time))

            eventp = []
            for i in range(self.N_neurons):
                eventp.append([])

            Vm = -1 * np.ones((self.N_neurons,self.learning_time))
            Vf = 0 * np.ones((self.N_neurons,self.learning_time))
            Vs = 0 * np.ones((self.N_neurons,self.learning_time))
            Vus = 0 * np.ones((self.N_neurons,self.learning_time))

            scale = 1
            scale_synapse = 0.1
            tauf = 1
            taus = 50
            tauus = 50 * 50
            dT = 1
            Erev=0.48
            Vthresh=2

            dec = 0
            cur_spk = [0,0,0,0]
            st_time = 0

            pattern_coeff = 0
            if pattern_id == 1:
                pattern_coeff = 1
            elif pattern_id == 2:
                pattern_coeff = -1

            actual_spike_ctr = [0, 0, 0, 0]
            for t in tqdm(range(2, self.learning_time)):
                one_hot_encoded = np.zeros(5)
                # WALK
                if pattern_id == 1:
                    one_hot_encoded[4] = 0
                # CRAWL
                else:
                    one_hot_encoded[4] = 1

                if t==10000:
                    dec = 1
                    st_time = t
                
                for cur_neu in range(self.N_neurons):
                    dVfdT = (Vm[cur_neu][t - 1] - Vf[cur_neu][t - 1]) / tauf
                    dVsdT = (Vm[cur_neu][t - 1] - Vs[cur_neu][t - 1]) / taus
                    dVusdT = (Vm[cur_neu][t - 1] - Vus[cur_neu][t - 1]) / tauus
                    
                    Vf[cur_neu][t] = Vf[cur_neu][t - 1] + (dVfdT * dT)
                    Vs[cur_neu][t] = Vs[cur_neu][t - 1] + (dVsdT * dT)
                    Vus[cur_neu][t] = Vus[cur_neu][t - 1] + (dVusdT * dT)
                    
                    F_N = alpha[0] * np.tanh((Vf[cur_neu][t - 1] - delta[0]) * scale)
                    S_P = alpha[1] * np.tanh((Vs[cur_neu][t - 1] - delta[1]) * scale)
                    S_N = alpha[2] * np.tanh((Vs[cur_neu][t - 1] - delta[2]) * scale)
                    US_N = alpha[3] * np.tanh((Vus[cur_neu][t - 1] - delta[3]) * scale)
                    
                    Isum = 0
                    for conn in range(self.N_neurons):
                        Isum += self.W[cur_neu][conn] * self.sf(Vs[conn][t], 2, 1)
                    
                    I_x = F_N + S_P + S_N + US_N
                    I_P = (Vm[cur_neu][t - 1] / self.R)
                    
                    dVmdT = (- I_P - I_x + I[cur_neu][t] + Isum) / Cm
                    Vm[cur_neu][t] = Vm[cur_neu][t - 1] + (dVmdT * dT)
                    
                    if Vm[cur_neu][t] > Vthresh and Vm[cur_neu][t-1] < Vthresh:
                        eventp[cur_neu].append(t)
                        actual_spike_ctr[cur_neu] += 1
                    
                    if actual_spike_ctr[cur_neu] >= 8:  # 8 spikes/burst
                        actual_spike_ctr[cur_neu] = 0
                        one_hot_encoded[cur_neu] = 1

                if dec == 1:
                    for cur_neu in range(int(self.N_neurons)):
                        for conn in range(self.N_neurons):
                            if conn!=cur_neu:
                                if (abs(ideal_spike_times[cur_neu][cur_spk[cur_neu]]-(eventp[cur_neu][-1]-st_time)%1600)>100) and ideal_spike_times[cur_neu][conn]==1:
                                    self.W[cur_neu][conn] = -0.5*pattern_coeff*self.sf(self.W2[cur_neu][conn]+self.train_STDP(0.1,((eventp[cur_neu][-1]-st_time)%1600),self.walk_ideal[cur_neu][cur_spk[cur_neu]],100),1,0)
                                elif pattern_curr_conn[cur_neu][conn]==0:
                                    self.W[cur_neu][conn] = 0.5*pattern_coeff*self.sf(self.W2[cur_neu][conn]+self.train_STDP(0.1,((eventp[cur_neu][-1]-st_time)%1600),self.walk_ideal[cur_neu][cur_spk[cur_neu]],100),1,0)
                        
                        cur_spk[cur_neu]+=1
                        if cur_spk[cur_neu] == 8:
                            cur_spk[cur_neu] = 0
                
                # MOTOR COMMANDS
                if not np.all(one_hot_encoded[:4] == 0):
                    data_str = ','.join(map(str, one_hot_encoded.astype(int))) + '\n'
                    ser.write(data_str.encode())

        return self.W
