import numpy as np
import time
import os

spikes_or_fr_file = os.path.join('./data', 'spikes_or_fr.txt')
spikes_not_fr_file = os.path.join('./data', 'spikes_not_fr.txt')
spikes_or_fl_file = os.path.join('./data', 'spikes_or_fl.txt')
spikes_not_fl_file = os.path.join('./data', 'spikes_not_fl.txt')
spikes_or_br_file = os.path.join('./data', 'spikes_or_br.txt')
spikes_not_br_file = os.path.join('./data', 'spikes_not_br.txt')
spikes_or_bl_file = os.path.join('./data', 'spikes_or_bl.txt')
spikes_not_bl_file = os.path.join('./data', 'spikes_not_bl.txt')

spikes_or_fr = []
spikes_not_fr = []
spikes_or_fl = []
spikes_not_fl = []
spikes_or_br = []
spikes_not_br = []
spikes_or_bl = []
spikes_not_bl = []


TMAX = 1000000

def sf(x,b,ds):
    k=b*(x-ds)
    return 1/(1+np.exp(-k))


def d_sf(asyn,x,b,ds):
  return asyn*((b*np.exp(b*(x-ds)))/((np.exp(b*(x-ds))+1)**2))


b=5

# Neurons (N number of neuron in first layer) (M number of neuron in second layer)
L1_neu, L2_neu = 4, 8
lif_temp=[0] * L2_neu
vth = 5

L1_v_thresh = 2.0
# L1_v_thresh = 1.2

# or and not gates
L2_or = np.zeros([L2_neu, TMAX])
L2_not = np.zeros([L2_neu, TMAX])


# weights from L1 to L2 (walk)
g_lif=np.zeros((L1_neu, L2_neu))
g_lif[0][0] = vth+0.01  # FRK
g_lif[0][1] = vth+0.01  # FRS
g_lif[3][2] = vth+0.01  # BLK
g_lif[3][3] = vth+0.01  # BLS
g_lif[1][4] = vth+0.01  # FLK
g_lif[1][5] = vth+0.01  # FLS
g_lif[2][6] = vth+0.01  # BRK
g_lif[2][7] = vth+0.01  # BRS

# network weights
gp_inhib = -0.3

asyn = np.array([[0, gp_inhib, gp_inhib, gp_inhib],
                [gp_inhib, 0, gp_inhib, gp_inhib],
                [gp_inhib, gp_inhib, 0, gp_inhib],
                [gp_inhib, gp_inhib, gp_inhib, 0]])


def set_weights(W):
    global asyn
    asyn = W


dsyn = np.array([[0, -1, -1, -1],
                 [-1, 0, -1, -1],
                 [-1, -1, 0, -1],
                 [-1, -1, -1, 0]])
# Iapp
Iapp = 1.7 * np.array([-1, -1, -1, -1])


### Neurons parameters ###
# [af-,as+,as-,aus+]
alpha = [-2, 2, -1.5, 4]
# [df-,ds+,ds-,dus+]
delta = [0, 0, -1.5, -1.5]

# Tf, Ts, Tus = 1, 5, 250
Tf, Ts, Tus = 1, 50, 2500

# vm1_0, vf1_0, vs1_0, vus1_0 = -1, 0, 0, 0
# vm2_0, vf2_0, vs2_0, vus2_0 = -1, 0, 0, 0
# vm3_0, vf3_0, vs3_0, vus3_0 = -1, 0, 0, 0
# vm4_0, vf4_0, vs4_0, vus4_0 = -1, 0, 0, 0
vm1_0, vf1_0, vs1_0, vus1_0 = -1.39888908, -1.39501677, -0.98051566, -1.53577068
vm2_0, vf2_0, vs2_0, vus2_0 = -2.43159863, -2.4364570, -2.46313154, -1.71510205 
vm3_0, vf3_0, vs3_0, vus3_0 = -3.01994828, -3.02413098, -2.84032527, -1.61383641
vm4_0, vf4_0, vs4_0, vus4_0 = 0.85359452,  0.91745326,  0.48567456, -1.66450763
S_0 = (vm1_0, vf1_0, vs1_0, vus1_0,
       vm2_0, vf2_0, vs2_0, vus2_0,
       vm3_0, vf3_0, vs3_0, vus3_0,
       vm4_0, vf4_0, vs4_0, vus4_0)


def f(S):
    vm1, vf1, vs1, vus1, vm2, vf2, vs2, vus2, vm3, vf3, vs3, vus3, vm4, vf4, vs4, vus4 = S
    Isyn1 = Iapp[0] + asyn[1][0] * sf(vs2, b, dsyn[1][0]) + asyn[2][0] * sf(vs3, b, dsyn[2][0]) + asyn[3][0] * sf(vs4,
                                                                                                                  b,
                                                                                                                  dsyn[
                                                                                                                      3][
                                                                                                                      0])
    Isyn2 = Iapp[1] + asyn[0][1] * sf(vs1, b, dsyn[0][1]) + asyn[2][1] * sf(vs3, b, dsyn[2][1]) + asyn[3][1] * sf(vs4,
                                                                                                                  b,
                                                                                                                  dsyn[
                                                                                                                      3][
                                                                                                                      1])
    Isyn3 = Iapp[2] + asyn[0][2] * sf(vs1, b, dsyn[0][2]) + asyn[1][2] * sf(vs2, b, dsyn[1][2]) + asyn[3][2] * sf(vs4,
                                                                                                                  b,
                                                                                                                  dsyn[
                                                                                                                      3][
                                                                                                                      2])
    Isyn4 = Iapp[3] + asyn[0][3] * sf(vs1, b, dsyn[0][3]) + asyn[1][3] * sf(vs2, b, dsyn[1][2]) + asyn[2][3] * sf(vs3,
                                                                                                                  b,
                                                                                                                  dsyn[
                                                                                                                      2][
                                                                                                                      3])


    return [-vm1 - alpha[0] * np.tanh(vf1 - delta[0]) - alpha[1] * np.tanh(vs1 - delta[1]) - alpha[2] * np.tanh(
        vs1 - delta[2]) - alpha[3] * np.tanh(vus1 - delta[3]) + Isyn1, (vm1 - vf1) / Tf, (vm1 - vs1) / Ts,
            (vm1 - vus1) / Tus,
            -vm2 - alpha[0] * np.tanh(vf2 - delta[0]) - alpha[1] * np.tanh(vs2 - delta[1]) - alpha[2] * np.tanh(
                vs2 - delta[2]) - alpha[3] * np.tanh(vus2 - delta[3]) + Isyn2, (vm2 - vf2) / Tf, (vm2 - vs2) / Ts,
            (vm2 - vus2) / Tus,
            -vm3 - alpha[0] * np.tanh(vf3 - delta[0]) - alpha[1] * np.tanh(vs3 - delta[1]) - alpha[2] * np.tanh(
                vs3 - delta[2]) - alpha[3] * np.tanh(vus3 - delta[3]) + Isyn3, (vm3 - vf3) / Tf, (vm3 - vs3) / Ts,
            (vm3 - vus3) / Tus,
            -vm4 - alpha[0] * np.tanh(vf4 - delta[0]) - alpha[1] * np.tanh(vs4 - delta[1]) - alpha[2] * np.tanh(
                vs4 - delta[2]) - alpha[3] * np.tanh(vus4 - delta[3]) + Isyn4, (vm4 - vf4) / Tf, (vm4 - vs4) / Ts,
            (vm4 - vus4) / Tus]


Tf_inv = 1 / Tf
Ts_inv = 1 / Ts
Tus_inv = 1 / Tus


def df(S):
    vm1, vf1, vs1, vus1, vm2, vf2, vs2, vus2, vm3, vf3, vs3, vus3, vm4, vf4, vs4, vus4 = S
    return np.array([[-1, -alpha[0] * (1 - np.tanh(vf1 - delta[0]) ** 2),
             -alpha[1] * (1 - np.tanh(vs1 - delta[1]) ** 2) - alpha[2] * (1 - np.tanh(vs1 - delta[2]) ** 2),
             -alpha[3] * (1 - np.tanh(vus1 - delta[3]) ** 2), 0, 0, d_sf(asyn[1][0], vs2, b, dsyn[1][0]), 0, 0, 0,
             d_sf(asyn[2][0], vs3, b, dsyn[2][0]), 0, 0, 0, d_sf(asyn[3][0], vs4, b, dsyn[3][0]), 0],
            [Tf_inv, -Tf_inv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [Ts_inv, 0, -Ts_inv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [Tus_inv, 0, 0, -Tus_inv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, d_sf(asyn[0][1], vs1, b, dsyn[0][1]), 0, -1, -alpha[0] * (1 - np.tanh(vf2 - delta[0]) ** 2),
             -alpha[1] * (1 - np.tanh(vs2 - delta[1]) ** 2) - alpha[2] * (1 - np.tanh(vs2 - delta[2]) ** 2),
             -alpha[3] * (1 - np.tanh(vus2 - delta[3]) ** 2), 0, 0, d_sf(asyn[2][1], vs3, b, dsyn[2][1]), 0, 0, 0,
             d_sf(asyn[3][1], vs4, b, dsyn[3][1]), 0],
            [0, 0, 0, 0, Tf_inv, -Tf_inv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, Ts_inv, 0, -Ts_inv, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, Tus_inv, 0, 0, -Tus_inv, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, d_sf(asyn[0][2], vs1, b, dsyn[0][2]), 0, 0, 0, d_sf(asyn[1][2], vs2, b, dsyn[1][2]), 0, -1,
             -alpha[0] * (1 - np.tanh(vf3 - delta[0]) ** 2),
             -alpha[1] * (1 - np.tanh(vs3 - delta[1]) ** 2) - alpha[2] * (1 - np.tanh(vs3 - delta[2]) ** 2),
             -alpha[3] * (1 - np.tanh(vus3 - delta[3]) ** 2), 0, 0, d_sf(asyn[3][2], vs4, b, dsyn[3][2]), 0],
            [0, 0, 0, 0, 0, 0, 0, 0, Tf_inv, -Tf_inv, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, Ts_inv, 0, -Ts_inv, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, Tus_inv, 0, 0, -Tus_inv, 0, 0, 0, 0],
            [0, 0, d_sf(asyn[0][3], vs1, b, dsyn[0][3]), 0, 0, 0, d_sf(asyn[1][3], vs2, b, dsyn[1][3]), 0, 0, 0,
             d_sf(asyn[2][3], vs3, b, dsyn[2][3]), 0, -1, -alpha[0] * (1 - np.tanh(vf4 - delta[0]) ** 2),
             -alpha[1] * (1 - np.tanh(vs4 - delta[1]) ** 2) - alpha[2] * (1 - np.tanh(vs4 - delta[2]) ** 2),
             -alpha[3] * (1 - np.tanh(vus4 - delta[3]) ** 2)],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Tf_inv, -Tf_inv, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Ts_inv, 0, -Ts_inv, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Tus_inv, 0, 0, -Tus_inv],
            ]).ravel()


def newton(f, Jf, x0, epsilon, max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x,y,z,...)=0.
    Df : function
        Jacobian of f(x,y,z,...).
    x0 : number
        Initial guess for a solution f(x,y,z,...)=0.
    epsilon : number
        Stopping criteria is norm(Df(x,y,z,...)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''

    xn = np.array(x0)
    delta = 1
    CF = False
    for n in range(0, max_iter):
        fxn = f(xn)

        if abs(delta) < epsilon:
            CF = True
            # print('Found solution after',n,'iterations.')
            return [xn, CF]
        
        Dfxn = Jf(xn)
        # Coovergence condition (if the norm of the ||delta||<epsilon)
        mat_mult = np.linalg.solve(Dfxn, fxn)
        delta = np.linalg.norm(mat_mult)

        # Det check
        if np.linalg.det(Dfxn) == 0:
            CF = False
            # print('Zero derivative. No solution found.')
            return [None, CF]

        # Calculation
        xn -= mat_mult
    print('Exceeded maximum iterations. No solution found.')
    return [None, CF]


def bdf_second(ser, pipe, f=f, y0=S_0, h=3.35, ti=0, tf=TMAX):
    while True:
        if pipe.poll():
            cmd = pipe.recv()
            if cmd == 'Start':
                print("Started network evolution")
                time.sleep(2.0)
                break

    # Initializing
    N=int((tf-ti)//h) # Number of steps 
    t=np.linspace(ti,tf,N) # Timestep vector
    y=np.zeros([int(len(y0)),N])
    L1_spk_out = np.zeros([L1_neu, N])
    for k in range(int(len(y0))):
        y[k][0]=y0[k]

    O=1 # for 2nd BDF 
    for i in range(0,O):
        # g, Next value function. 
        def g(S):
            # Next value of state variables, vmn = vm(k+1)
            vmn1, vfn1, vsn1, vusn1, vmn2, vfn2, vsn2, vusn2, vmn3, vfn3, vsn3, vusn3, vmn4, vfn4, vsn4, vusn4 = S
            # Next value of state variable, fvmn = f(t[i+1],S) 
            fvmn1, fvfn1, fvsn1, fvusn1, fvmn2, fvfn2, fvsn2, fvusn2, fvmn3, fvfn3, fvsn3, fvusn3, fvmn4, fvfn4, fvsn4, fvusn4 = f(S)
            
            return np.array([vmn1-(h*fvmn1)-y[0][i],vfn1-(h*fvfn1)-y[1][i],vsn1-(h*fvsn1)-y[2][i],vusn1-(h*fvusn1)-y[3][i],
                             vmn2-(h*fvmn2)-y[4][i],vfn2-(h*fvfn2)-y[5][i],vsn2-(h*fvsn2)-y[6][i],vusn2-(h*fvusn2)-y[7][i],
                             vmn3-(h*fvmn3)-y[8][i],vfn3-(h*fvfn3)-y[9][i],vsn3-(h*fvsn3)-y[10][i],vusn3-(h*fvusn3)-y[11][i],
                             vmn4-(h*fvmn4)-y[12][i],vfn4-(h*fvfn4)-y[13][i],vsn4-(h*fvsn4)-y[14][i],vusn4-(h*fvusn4)-y[15][i]])
        
        def Jg(S):
            # next value of state variables, vmn = vm(n+1)
            Jf = df(S).reshape(16, 16)
            res = []
            for m in range(Jf.shape[0]):
                temp = []
                for n in range(Jf.shape[1]):
                    if m == n:
                        temp.append(1 - h * Jf[m][n])
                    else:
                        temp.append(-h * Jf[m][n])
                res.append(temp)

            return np.array(res)
        
        # initial condition for Newthon's method
        x0=[y[k][i] for k in range(int(len(y0)))]
        # x0=[y[0][i],y[1][i],y[2][i],y[3][i]]
        root=newton(g,Jg,x0,1e-10,20)
        # root[1] = bolean flag to make sure that the newthon was successful.
        if root[1]:
            for k in range(int(len(y0))):
                y[k][i+1] = root[0][k]
        else:
            print('No solution found')
    #============================================================================================================================

    commands = np.zeros(9)
    for i in range(0,N-2):
        # commands to arduino (9th element for gait selection)
        commands[:8] = 0

        # 3: walk, 2: crawl
        if pipe.poll():
            cmd = pipe.recv()
            if cmd == 'Crawl':
                g_lif=np.zeros((L1_neu, L2_neu))
                # N1 + N2
                g_lif[0][0] = vth + 0.01  # FRK
                g_lif[0][1] = vth + 0.01  # FRS
                g_lif[3][0] = vth + 0.01  # FRK
                g_lif[3][1] = vth + 0.01  # FRS
                # N2 + N3
                g_lif[3][2] = vth + 0.01  # BLK
                g_lif[3][3] = vth + 0.01  # BLS
                g_lif[1][2] = vth + 0.01  # BLK
                g_lif[1][3] = vth + 0.01  # BLS
                # N3 + N4
                g_lif[1][4] = vth + 0.01  # FLK
                g_lif[1][5] = vth + 0.01  # FLS
                g_lif[2][4] = vth + 0.01  # FLK
                g_lif[2][5] = vth + 0.01  # FLS
                # N4 + N1
                g_lif[2][6] = vth + 0.01  # BRK
                g_lif[2][7] = vth + 0.01  # BRS
                g_lif[0][6] = vth + 0.01  # BRK
                g_lif[0][7] = vth + 0.01  # BRS
                print("Started crawl")
                commands[8] = 2
                # time.sleep(2.0)

            elif cmd == 'Walk':
                g_lif=np.zeros((L1_neu, L2_neu))
                g_lif[0][0] = vth + 0.01  # FRK
                g_lif[0][1] = vth + 0.01  # FRS
                g_lif[3][2] = vth + 0.01  # BLK
                g_lif[3][3] = vth + 0.01  # BLS
                g_lif[1][4] = vth + 0.01  # FLK
                g_lif[1][5] = vth + 0.01  # FLS
                g_lif[2][6] = vth + 0.01  # BRK
                g_lif[2][7] = vth + 0.01  # BRS
                print("Started walk")
                commands[8] = 3
                # time.sleep(2.0)

                # else:   # idle
                #     g_lif=np.zeros((L1_neu, L2_neu))
                #     commands[8] = -1

        # g, Next value function. 
        def g(S):
            # Next value of state variables, vmn = vm(k+1)
            vmn1, vfn1, vsn1, vusn1, vmn2, vfn2, vsn2, vusn2, vmn3, vfn3, vsn3, vusn3, vmn4, vfn4, vsn4, vusn4 = S
            # Next value of state variable, fvmn = f(t[i+1],S) 
            fvmn1, fvfn1, fvsn1, fvusn1, fvmn2, fvfn2, fvsn2, fvusn2, fvmn3, fvfn3, fvsn3, fvusn3, fvmn4, fvfn4, fvsn4, fvusn4 = f(S)
            cf1 = (2/3)
            cf2 = (4/3)
            cf3 = (1/3)
            return np.array([vmn1-(cf1*h*fvmn1)-cf2*y[0][i+1]+cf3*y[0][i],vfn1-(cf1*h*fvfn1)-cf2*y[1][i+1]+cf3*y[1][i],vsn1-(cf1*h*fvsn1)-cf2*y[2][i+1]+cf3*y[2][i],vusn1-(cf1*h*fvusn1)-cf2*y[3][i+1]+cf3*y[3][i],
                             vmn2-(cf1*h*fvmn2)-cf2*y[4][i+1]+cf3*y[4][i],vfn2-(cf1*h*fvfn2)-cf2*y[5][i+1]+cf3*y[5][i],vsn2-(cf1*h*fvsn2)-cf2*y[6][i+1]+cf3*y[6][i],vusn2-(cf1*h*fvusn2)-cf2*y[7][i+1]+cf3*y[7][i],
                             vmn3-(cf1*h*fvmn3)-cf2*y[8][i+1]+cf3*y[8][i],vfn3-(cf1*h*fvfn3)-cf2*y[9][i+1]+cf3*y[9][i],vsn3-(cf1*h*fvsn3)-cf2*y[10][i+1]+cf3*y[10][i],vusn3-(cf1*h*fvusn3)-cf2*y[11][i+1]+cf3*y[11][i],
                             vmn4-(cf1*h*fvmn4)-cf2*y[12][i+1]+cf3*y[12][i],vfn4-(cf1*h*fvfn4)-cf2*y[13][i+1]+cf3*y[13][i],vsn4-(cf1*h*fvsn4)-cf2*y[14][i+1]+cf3*y[14][i],vusn4-(cf1*h*fvusn4)-cf2*y[15][i+1]+cf3*y[15][i]])
            

        def Jg(S):
            # next value of state variables, vmn = vm(n+1)
            CF = 2 / 3

            Jf = df(S).reshape(16, 16)
            res = []
            for m in range(Jf.shape[0]):
                temp = []
                for n in range(Jf.shape[1]):
                    if m == n:
                        temp.append(1 - CF * h * Jf[m][n])
                    else:
                        temp.append(-CF * h * Jf[m][n])
                res.append(temp)

            return np.array(res)
        
        # initial condition for Newthon's method
        x0=[y[k][i+1] for k in range(int(len(y0)))]
        #x0=[y[0][i],y[1][i],y[2][i],y[3][i],y[4][i],y[5][i],y[6][i],y[7][i],y[8][i],y[9][i],y[10][i],y[11][i],y[12][i],y[13][i],y[14][i],y[15][i]]
        root=newton(g,Jg,x0,1e-3,10)
        # root[1] = bolean flag to make sure that the newthon was successful.
        if root[1]:
            for k in range(int(len(y0))):
                y[k][i+2] = root[0][k]
                # thresholding to send spikes to L2
                if k % 4 == 0 and y[k][i + 2] > L1_v_thresh:
                    L1_spk_out[k//4][i+2] = 1

            # L2 LIF calculation
            for l in range(L2_neu):  # L2
                for k in range(L1_neu): # L1
                    lif_temp[l] += g_lif[k][l] * L1_spk_out[k][i]
                if lif_temp[l] > vth:
                    # or gate
                    L2_or[l][i] = 1
                    lif_temp[l] = 0
                    # not gates
                    for m in range(L2_neu):
                        if m == l or np.array_equal(g_lif[:, m], g_lif[:, l]):
                            L2_not[m][i] = 0
                        else:
                            L2_not[m][i] = 1

                else:
                    # or gate
                    L2_or[l][i] = lif_temp[l]
                    # not gates
                    for m in range(L2_neu):
                        if m == l or L2_not[m][i] == 1:
                            continue
                        L2_not[m][i] = 0

            # send motor commands
            # 0: FRK, 1: FRS, 2: BLK, 3: BLS, 4: FLK, 5: FLS, 6: BRK, 7: BRS (L2 mapping)            
            if L2_or[5][i] == 1:   # FLS_burst
                commands[0] = 1
                spikes_or_fl.append(i)
            elif L2_not[5][i] == 1:     # FLS_non_burst
                commands[0] = 1
                spikes_not_fl.append(i)

            if L2_or[1][i] == 1:    # FRS_burst
                commands[1] = 1
                spikes_or_fr.append(i)
            elif L2_not[1][i] == 1:     # FRS_non_burst
                commands[1] = 1
                spikes_not_fr.append(i)

            if L2_or[7][i] == 1:    # BRS_burst
                commands[2] = 1
                spikes_or_br.append(i)
            elif L2_not[7][i] == 1:     # BRS_non_burst
                commands[2] = 1
                spikes_not_br.append(i)

            if L2_or[3][i] == 1:   # BLS_burst
                commands[3] = 1
                spikes_or_bl.append(i)
            elif L2_not[3][i]== 1:     # BLS_non_burst
                commands[3] = 1
                spikes_not_bl.append(i)

            if L2_or[4][i] == 1:  # FLK_burst
                commands[4] = 1
            elif L2_not[4][i]== 1:  # FLK_non_burst
                commands[4] = 1

            if L2_or[0][i] == 1:  # FRK_burst
                commands[5] = 1
            elif L2_not[0][i] == 1:  # FRK_non_burst
                commands[5] = 1

            if L2_or[6][i] == 1:  # BRK_burst
                commands[6] = 1
            elif L2_not[6][i] == 1:  # BRK_non_burst
                commands[6] = 1

            if L2_or[2][i] == 1:  # BLK_burst
                commands[7] = 1
            elif L2_not[2][i] == 1:  # BLK_non_burst
                commands[7] = 1

            # send commands to Arduino
            if not np.all(commands[:8] == 0):
                data_str = ','.join(map(str, commands.astype(int))) + '\n'
                ser.write(data_str.encode())
                # print("mtr cmds = ", data_str)
                # time.sleep(0.005)

        else:
            print('No solution found')


        if i % 1000 == 0:
            # save spikes data
            np.savetxt(spikes_or_fr_file, spikes_or_fr)
            np.savetxt(spikes_not_fr_file, spikes_not_fr)
            np.savetxt(spikes_or_fl_file, spikes_or_fl)
            np.savetxt(spikes_not_fl_file, spikes_not_fl)
            np.savetxt(spikes_or_br_file, spikes_or_br)
            np.savetxt(spikes_not_br_file, spikes_not_br)
            np.savetxt(spikes_or_bl_file, spikes_or_bl)
            np.savetxt(spikes_not_bl_file, spikes_not_bl)
