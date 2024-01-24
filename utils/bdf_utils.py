import numpy as np

def sf(x,b,ds):
    k=b*(x-ds)
    return 1/(1+np.exp(-k))

def d_sf(asyn,x,b,ds):
  return asyn*((b*np.exp(b*(x-ds)))/((np.exp(b*(x-ds))+1)**2))
b=2

# network weights
asyn = np.array([[0,0.3,0.3,-0.3],
               [0.3,0,0.3,-0.3],
               [0.3,0.3,0,-0.3],
               [-0.3,-0.3,-0.3,0]])

def set_weights(W):
    global asyn
    asyn = W


dsyn=np.array([[0,-1,-1,-1],
               [-1,0,-1,-1],
               [-1,-1,0,-1],
               [-1,-1,-1,0]])
# Iapp
Iapp = 1.5*np.array([-1,-1,-1,-1])

#plt.style.use(['custom','no-latex'])

### Neurons parameters ###
# [af-,as+,as-,aus+]
alpha=[-2,2,-1.5,1.5]
# [df-,ds+,ds-,dus+]
delta=[0,0,-1.5,-1.5]

Tf, Ts, Tus = 1, 5, 250

# dsdt= f(t,vm,vf,vs,vus)
def f(S):
    vm1, vf1, vs1, vus1, vm2, vf2, vs2, vus2, vm3, vf3, vs3, vus3, vm4, vf4, vs4, vus4 = S
    Isyn1=Iapp[0]+asyn[1][0]*sf(vs2,b,dsyn[1][0])+asyn[2][0]*sf(vs3,b,dsyn[2][0])+asyn[3][0]*sf(vs4,b,dsyn[3][0]) 
    Isyn2=Iapp[1]+asyn[0][1]*sf(vs1,b,dsyn[0][1])+asyn[2][1]*sf(vs3,b,dsyn[2][1])+asyn[3][1]*sf(vs4,b,dsyn[3][1])
    Isyn3=Iapp[2]+asyn[0][2]*sf(vs1,b,dsyn[0][2])+asyn[1][2]*sf(vs2,b,dsyn[1][2])+asyn[3][2]*sf(vs4,b,dsyn[3][2])
    Isyn4=Iapp[3]+asyn[0][3]*sf(vs1,b,dsyn[0][3])+asyn[1][3]*sf(vs2,b,dsyn[1][2])+asyn[2][3]*sf(vs3,b,dsyn[2][3])
    return [-vm1-alpha[0]*np.tanh(vf1-delta[0])-alpha[1]*np.tanh(vs1-delta[1])-alpha[2]*np.tanh(vs1-delta[2])-alpha[3]*np.tanh(vus1-delta[3])+Isyn1,(vm1-vf1)/Tf,(vm1-vs1)/Ts,(vm1-vus1)/Tus,
            -vm2-alpha[0]*np.tanh(vf2-delta[0])-alpha[1]*np.tanh(vs2-delta[1])-alpha[2]*np.tanh(vs2-delta[2])-alpha[3]*np.tanh(vus2-delta[3])+Isyn2,(vm2-vf2)/Tf,(vm2-vs2)/Ts,(vm2-vus2)/Tus,
            -vm3-alpha[0]*np.tanh(vf3-delta[0])-alpha[1]*np.tanh(vs3-delta[1])-alpha[2]*np.tanh(vs3-delta[2])-alpha[3]*np.tanh(vus3-delta[3])+Isyn3,(vm3-vf3)/Tf,(vm3-vs3)/Ts,(vm3-vus3)/Tus,
            -vm4-alpha[0]*np.tanh(vf4-delta[0])-alpha[1]*np.tanh(vs4-delta[1])-alpha[2]*np.tanh(vs4-delta[2])-alpha[3]*np.tanh(vus4-delta[3])+Isyn4,(vm4-vf4)/Tf,(vm4-vs4)/Ts,(vm4-vus4)/Tus]

def df(S):
    vm1, vf1, vs1, vus1, vm2, vf2, vs2, vus2, vm3, vf3, vs3, vus3, vm4, vf4, vs4, vus4 = S
    return [[-1,-alpha[0]*(1-np.tanh(vf1-delta[0])**2),-alpha[1]*(1-np.tanh(vs1-delta[1])**2)-alpha[2]*(1-np.tanh(vs1-delta[2])**2),-alpha[3]*(1-np.tanh(vus1-delta[3])**2),0,0,d_sf(asyn[1][0],vs2,b,dsyn[1][0]),0,0,0,d_sf(asyn[2][0],vs3,b,dsyn[2][0]),0,0,0,d_sf(asyn[3][0],vs4,b,dsyn[3][0]),0],
            [(1/Tf),(-1/Tf),0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [(1/Ts),0,(-1/Ts),0,0,0,0,0,0,0,0,0,0,0,0,0],
            [(1/Tus),0,0,(-1/Tus),0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,d_sf(asyn[0][1],vs1,b,dsyn[0][1]),0,-1,-alpha[0]*(1-np.tanh(vf2-delta[0])**2),-alpha[1]*(1-np.tanh(vs2-delta[1])**2)-alpha[2]*(1-np.tanh(vs2-delta[2])**2),-alpha[3]*(1-np.tanh(vus2-delta[3])**2),0,0,d_sf(asyn[2][1],vs3,b,dsyn[2][1]),0,0,0,d_sf(asyn[3][1],vs4,b,dsyn[3][1]),0],
            [0,0,0,0,(1/Tf),(-1/Tf),0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,(1/Ts),0,(-1/Ts),0,0,0,0,0,0,0,0,0],
            [0,0,0,0,(1/Tus),0,0,(-1/Tus),0,0,0,0,0,0,0,0],
            [0,0,d_sf(asyn[0][2],vs1,b,dsyn[0][2]),0,0,0,d_sf(asyn[1][2],vs2,b,dsyn[1][2]),0,-1,-alpha[0]*(1-np.tanh(vf3-delta[0])**2),-alpha[1]*(1-np.tanh(vs3-delta[1])**2)-alpha[2]*(1-np.tanh(vs3-delta[2])**2),-alpha[3]*(1-np.tanh(vus3-delta[3])**2),0,0,d_sf(asyn[3][2],vs4,b,dsyn[3][2]),0],
            [0,0,0,0,0,0,0,0,(1/Tf),(-1/Tf),0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,(1/Ts),0,(-1/Ts),0,0,0,0,0],
            [0,0,0,0,0,0,0,0,(1/Tus),0,0,(-1/Tus),0,0,0,0],
            [0,0,d_sf(asyn[0][3],vs1,b,dsyn[0][3]),0,0,0,d_sf(asyn[1][3],vs2,b,dsyn[1][3]),0,0,0,d_sf(asyn[2][3],vs3,b,dsyn[2][3]),0,-1,-alpha[0]*(1-np.tanh(vf4-delta[0])**2),-alpha[1]*(1-np.tanh(vs4-delta[1])**2)-alpha[2]*(1-np.tanh(vs4-delta[2])**2),-alpha[3]*(1-np.tanh(vus4-delta[3])**2)],
            [0,0,0,0,0,0,0,0,0,0,0,0,(1/Tf),(-1/Tf),0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,(1/Ts),0,(-1/Ts),0],
            [0,0,0,0,0,0,0,0,0,0,0,0,(1/Tus),0,0,(-1/Tus)],
            ]

def newton(f,Jf,x0,epsilon,max_iter):
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
    delta=1
    CF=False
    for n in range(0,max_iter):
        fxn = f(xn)
        
        if abs(delta) < epsilon:
            CF=True
            # print('Found solution after',n,'iterations.')
            return [xn,CF]
        Dfxn = Jf(xn)
        # Coovergence condition (if the norm of the ||delta||<epsilon)
        delta=np.linalg.norm(np.matmul(np.linalg.inv(Dfxn),fxn))
        
        # Det check
        if np.linalg.det(Dfxn) == 0:
            CF=False
            # print('Zero derivative. No solution found.')
            return [None,CF]
        
        # Calculation
        xn = xn - np.matmul(np.linalg.inv(Dfxn),fxn)
    print('Exceeded maximum iterations. No solution found.')
    return [None,CF]

def bdf_second(f, y0, h, ti,tf):
    # Initializing
    N=int((tf-ti)//h) # Number of steps 
    t=np.linspace(ti,tf,N) # Timestep vector
    y=np.zeros([int(len(y0)),N])
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
            Jf=df(S)
            return np.array([[1-h*Jf[0][0],-h*Jf[0][1],-h*Jf[0][2],-h*Jf[0][3],-h*Jf[0][4],-h*Jf[0][5],-h*Jf[0][6],-h*Jf[0][7],-h*Jf[0][8],-h*Jf[0][9],-h*Jf[0][10],-h*Jf[0][11],-h*Jf[0][12],-h*Jf[0][13],-h*Jf[0][14],-h*Jf[0][15]],
                             [-h*Jf[1][0],1-h*Jf[1][1],-h*Jf[1][2],-h*Jf[1][3],-h*Jf[1][4],-h*Jf[1][5],-h*Jf[1][6],-h*Jf[1][7],-h*Jf[1][8],-h*Jf[1][9],-h*Jf[1][10],-h*Jf[1][11],-h*Jf[1][12],-h*Jf[1][13],-h*Jf[1][14],-h*Jf[1][15]],
                             [-h*Jf[2][0],-h*Jf[2][1],1-h*Jf[2][2],-h*Jf[2][3],-h*Jf[2][4],-h*Jf[2][5],-h*Jf[2][6],-h*Jf[2][7],-h*Jf[2][8],-h*Jf[2][9],-h*Jf[2][10],-h*Jf[2][11],-h*Jf[2][12],-h*Jf[2][13],-h*Jf[2][14],-h*Jf[2][15]],
                             [-h*Jf[3][0],-h*Jf[3][1],-h*Jf[3][2],1-h*Jf[3][3],-h*Jf[3][4],-h*Jf[3][5],-h*Jf[3][6],-h*Jf[3][7],-h*Jf[3][8],-h*Jf[3][9],-h*Jf[3][10],-h*Jf[3][11],-h*Jf[3][12],-h*Jf[3][13],-h*Jf[3][14],-h*Jf[3][15]],
                             [-h*Jf[4][0],-h*Jf[4][1],-h*Jf[4][2],-h*Jf[4][3],1-h*Jf[4][4],-h*Jf[4][5],-h*Jf[4][6],-h*Jf[4][7],-h*Jf[4][8],-h*Jf[4][9],-h*Jf[4][10],-h*Jf[4][11],-h*Jf[4][12],-h*Jf[4][13],-h*Jf[4][14],-h*Jf[4][15]],
                             [-h*Jf[5][0],-h*Jf[5][1],-h*Jf[5][2],-h*Jf[5][3],-h*Jf[5][4],1-h*Jf[5][5],-h*Jf[5][6],-h*Jf[5][7],-h*Jf[5][8],-h*Jf[5][9],-h*Jf[5][10],-h*Jf[5][11],-h*Jf[5][12],-h*Jf[5][13],-h*Jf[5][14],-h*Jf[5][15]],
                             [-h*Jf[6][0],-h*Jf[6][1],-h*Jf[6][2],-h*Jf[6][3],-h*Jf[6][4],-h*Jf[6][5],1-h*Jf[6][6],-h*Jf[6][7],-h*Jf[6][8],-h*Jf[6][9],-h*Jf[6][10],-h*Jf[6][11],-h*Jf[6][12],-h*Jf[6][13],-h*Jf[6][14],-h*Jf[6][15]],
                             [-h*Jf[7][0],-h*Jf[7][1],-h*Jf[7][2],-h*Jf[7][3],-h*Jf[7][4],-h*Jf[7][5],-h*Jf[7][6],1-h*Jf[7][7],-h*Jf[7][8],-h*Jf[7][9],-h*Jf[7][10],-h*Jf[7][11],-h*Jf[7][12],-h*Jf[7][13],-h*Jf[7][14],-h*Jf[7][15]],
                             [-h*Jf[8][0],-h*Jf[8][1],-h*Jf[8][2],-h*Jf[8][3],-h*Jf[8][4],-h*Jf[8][5],-h*Jf[8][6],-h*Jf[8][7],1-h*Jf[8][8],-h*Jf[8][9],-h*Jf[8][10],-h*Jf[8][11],-h*Jf[8][12],-h*Jf[8][13],-h*Jf[8][14],-h*Jf[8][15]],
                             [-h*Jf[9][0],-h*Jf[9][1],-h*Jf[9][2],-h*Jf[9][3],-h*Jf[9][4],-h*Jf[9][5],-h*Jf[9][6],-h*Jf[9][7],-h*Jf[9][8],1-h*Jf[9][9],-h*Jf[9][10],-h*Jf[9][11],-h*Jf[9][12],-h*Jf[9][13],-h*Jf[9][14],-h*Jf[9][15]],
                             [-h*Jf[10][0],-h*Jf[10][1],-h*Jf[10][2],-h*Jf[10][3],-h*Jf[10][4],-h*Jf[10][5],-h*Jf[10][6],-h*Jf[10][7],-h*Jf[10][8],-h*Jf[10][9],1-h*Jf[10][10],-h*Jf[10][11],-h*Jf[10][12],-h*Jf[10][13],-h*Jf[10][14],-h*Jf[10][15]],
                             [-h*Jf[11][0],-h*Jf[11][1],-h*Jf[11][2],-h*Jf[11][3],-h*Jf[11][4],-h*Jf[11][5],-h*Jf[11][6],-h*Jf[11][7],-h*Jf[11][8],-h*Jf[11][9],-h*Jf[11][10],1-h*Jf[11][11],-h*Jf[11][12],-h*Jf[11][13],-h*Jf[11][14],-h*Jf[11][15]],
                             [-h*Jf[12][0],-h*Jf[12][1],-h*Jf[12][2],-h*Jf[12][3],-h*Jf[12][4],-h*Jf[12][5],-h*Jf[12][6],-h*Jf[12][7],-h*Jf[12][8],-h*Jf[12][9],-h*Jf[12][10],-h*Jf[12][11],1-h*Jf[12][12],-h*Jf[12][13],-h*Jf[12][14],-h*Jf[12][15]],
                             [-h*Jf[13][0],-h*Jf[13][1],-h*Jf[13][2],-h*Jf[13][3],-h*Jf[13][4],-h*Jf[13][5],-h*Jf[13][6],-h*Jf[13][7],-h*Jf[13][8],-h*Jf[13][9],-h*Jf[13][10],-h*Jf[13][11],-h*Jf[13][12],1-h*Jf[13][13],-h*Jf[13][14],-h*Jf[13][15]],
                             [-h*Jf[14][0],-h*Jf[14][1],-h*Jf[14][2],-h*Jf[14][3],-h*Jf[14][4],-h*Jf[14][5],-h*Jf[14][6],-h*Jf[14][7],-h*Jf[14][8],-h*Jf[14][9],-h*Jf[14][10],-h*Jf[14][11],-h*Jf[14][12],-h*Jf[14][13],1-h*Jf[14][14],-h*Jf[14][15]],
                             [-h*Jf[15][0],-h*Jf[15][1],-h*Jf[15][2],-h*Jf[15][3],-h*Jf[15][4],-h*Jf[15][5],-h*Jf[15][6],-h*Jf[15][7],-h*Jf[15][8],-h*Jf[15][9],-h*Jf[15][10],-h*Jf[15][11],-h*Jf[15][12],-h*Jf[15][13],-h*Jf[15][14],1-h*Jf[15][15]]])
        
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


    for i in range(0,N-2):
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
            # vmn,vfn,vsn,vusn=S
            # return np.array([vmn-h*f(t[i+1],S)-y[i],vfn-h*f(t[i+1],S)-y[i],vsn-h*f(t[i+1],S)-y[i],vusn-h*f(t[i+1],S)-y[i]])
            #vmn1-((2/3)*h*fvmn1)-(4/3)*y[0][i+1]+(1/3)*y[0][i],vfn1-((2/3)*h*fvfn1)-(4/3)*y[1][i+1]+(1/3)*y[1][i],vsn1-((2/3)*h*fvsn1)-(4/3)*y[2][i+1]+(1/3)*y[2][i],vusn1-((2/3)*h*fvusn1)-(4/3)*y[3][i+1]+(1/3)*y[3][i],
        
        def Jg(S):
            # next value of state variables, vmn = vm(n+1)
            Jf=df(S)
            CF=2/3
            return np.array([[1-CF*h*Jf[0][0],-CF*h*Jf[0][1],-CF*h*Jf[0][2],-CF*h*Jf[0][3],-CF*h*Jf[0][4],-CF*h*Jf[0][5],-CF*h*Jf[0][6],-CF*h*Jf[0][7],-CF*h*Jf[0][8],-CF*h*Jf[0][9],-CF*h*Jf[0][10],-CF*h*Jf[0][11],-CF*h*Jf[0][12],-CF*h*Jf[0][13],-CF*h*Jf[0][14],-CF*h*Jf[0][15]],
                             [-CF*h*Jf[1][0],1-CF*h*Jf[1][1],-CF*h*Jf[1][2],-CF*h*Jf[1][3],-CF*h*Jf[1][4],-CF*h*Jf[1][5],-CF*h*Jf[1][6],-CF*h*Jf[1][7],-CF*h*Jf[1][8],-CF*h*Jf[1][9],-CF*h*Jf[1][10],-CF*h*Jf[1][11],-CF*h*Jf[1][12],-CF*h*Jf[1][13],-CF*h*Jf[1][14],-CF*h*Jf[1][15]],
                             [-CF*h*Jf[2][0],-CF*h*Jf[2][1],1-CF*h*Jf[2][2],-CF*h*Jf[2][3],-CF*h*Jf[2][4],-CF*h*Jf[2][5],-CF*h*Jf[2][6],-CF*h*Jf[2][7],-CF*h*Jf[2][8],-CF*h*Jf[2][9],-CF*h*Jf[2][10],-CF*h*Jf[2][11],-CF*h*Jf[2][12],-CF*h*Jf[2][13],-CF*h*Jf[2][14],-CF*h*Jf[2][15]],
                             [-CF*h*Jf[3][0],-CF*h*Jf[3][1],-CF*h*Jf[3][2],1-CF*h*Jf[3][3],-CF*h*Jf[3][4],-CF*h*Jf[3][5],-CF*h*Jf[3][6],-CF*h*Jf[3][7],-CF*h*Jf[3][8],-CF*h*Jf[3][9],-CF*h*Jf[3][10],-CF*h*Jf[3][11],-CF*h*Jf[3][12],-CF*h*Jf[3][13],-CF*h*Jf[3][14],-CF*h*Jf[3][15]],
                             [-CF*h*Jf[4][0],-CF*h*Jf[4][1],-CF*h*Jf[4][2],-CF*h*Jf[4][3],1-CF*h*Jf[4][4],-CF*h*Jf[4][5],-CF*h*Jf[4][6],-CF*h*Jf[4][7],-CF*h*Jf[4][8],-CF*h*Jf[4][9],-CF*h*Jf[4][10],-CF*h*Jf[4][11],-CF*h*Jf[4][12],-CF*h*Jf[4][13],-CF*h*Jf[4][14],-CF*h*Jf[4][15]],
                             [-CF*h*Jf[5][0],-CF*h*Jf[5][1],-CF*h*Jf[5][2],-CF*h*Jf[5][3],-CF*h*Jf[5][4],1-CF*h*Jf[5][5],-CF*h*Jf[5][6],-CF*h*Jf[5][7],-CF*h*Jf[5][8],-CF*h*Jf[5][9],-CF*h*Jf[5][10],-CF*h*Jf[5][11],-CF*h*Jf[5][12],-CF*h*Jf[5][13],-CF*h*Jf[5][14],-CF*h*Jf[5][15]],
                             [-CF*h*Jf[6][0],-CF*h*Jf[6][1],-CF*h*Jf[6][2],-CF*h*Jf[6][3],-CF*h*Jf[6][4],-CF*h*Jf[6][5],1-CF*h*Jf[6][6],-CF*h*Jf[6][7],-CF*h*Jf[6][8],-CF*h*Jf[6][9],-CF*h*Jf[6][10],-CF*h*Jf[6][11],-CF*h*Jf[6][12],-CF*h*Jf[6][13],-CF*h*Jf[6][14],-CF*h*Jf[6][15]],
                             [-CF*h*Jf[7][0],-CF*h*Jf[7][1],-CF*h*Jf[7][2],-CF*h*Jf[7][3],-CF*h*Jf[7][4],-CF*h*Jf[7][5],-CF*h*Jf[7][6],1-CF*h*Jf[7][7],-CF*h*Jf[7][8],-CF*h*Jf[7][9],-CF*h*Jf[7][10],-CF*h*Jf[7][11],-CF*h*Jf[7][12],-CF*h*Jf[7][13],-CF*h*Jf[7][14],-CF*h*Jf[7][15]],
                             [-CF*h*Jf[8][0],-CF*h*Jf[8][1],-CF*h*Jf[8][2],-CF*h*Jf[8][3],-CF*h*Jf[8][4],-CF*h*Jf[8][5],-CF*h*Jf[8][6],-CF*h*Jf[8][7],1-CF*h*Jf[8][8],-CF*h*Jf[8][9],-CF*h*Jf[8][10],-CF*h*Jf[8][11],-CF*h*Jf[8][12],-CF*h*Jf[8][13],-CF*h*Jf[8][14],-CF*h*Jf[8][15]],
                             [-CF*h*Jf[9][0],-CF*h*Jf[9][1],-CF*h*Jf[9][2],-CF*h*Jf[9][3],-CF*h*Jf[9][4],-CF*h*Jf[9][5],-CF*h*Jf[9][6],-CF*h*Jf[9][7],-CF*h*Jf[9][8],1-CF*h*Jf[9][9],-CF*h*Jf[9][10],-CF*h*Jf[9][11],-CF*h*Jf[9][12],-CF*h*Jf[9][13],-CF*h*Jf[9][14],-CF*h*Jf[9][15]],
                             [-CF*h*Jf[10][0],-CF*h*Jf[10][1],-CF*h*Jf[10][2],-CF*h*Jf[10][3],-CF*h*Jf[10][4],-CF*h*Jf[10][5],-CF*h*Jf[10][6],-CF*h*Jf[10][7],-CF*h*Jf[10][8],-CF*h*Jf[10][9],1-CF*h*Jf[10][10],-CF*h*Jf[10][11],-CF*h*Jf[10][12],-CF*h*Jf[10][13],-CF*h*Jf[10][14],-CF*h*Jf[10][15]],
                             [-CF*h*Jf[11][0],-CF*h*Jf[11][1],-CF*h*Jf[11][2],-CF*h*Jf[11][3],-CF*h*Jf[11][4],-CF*h*Jf[11][5],-CF*h*Jf[11][6],-CF*h*Jf[11][7],-CF*h*Jf[11][8],-CF*h*Jf[11][9],-CF*h*Jf[11][10],1-CF*h*Jf[11][11],-CF*h*Jf[11][12],-CF*h*Jf[11][13],-CF*h*Jf[11][14],-CF*h*Jf[11][15]],
                             [-CF*h*Jf[12][0],-CF*h*Jf[12][1],-CF*h*Jf[12][2],-CF*h*Jf[12][3],-CF*h*Jf[12][4],-CF*h*Jf[12][5],-CF*h*Jf[12][6],-CF*h*Jf[12][7],-CF*h*Jf[12][8],-CF*h*Jf[12][9],-CF*h*Jf[12][10],-CF*h*Jf[12][11],1-CF*h*Jf[12][12],-CF*h*Jf[12][13],-CF*h*Jf[12][14],-CF*h*Jf[12][15]],
                             [-CF*h*Jf[13][0],-CF*h*Jf[13][1],-CF*h*Jf[13][2],-CF*h*Jf[13][3],-CF*h*Jf[13][4],-CF*h*Jf[13][5],-CF*h*Jf[13][6],-CF*h*Jf[13][7],-CF*h*Jf[13][8],-CF*h*Jf[13][9],-CF*h*Jf[13][10],-CF*h*Jf[13][11],-CF*h*Jf[13][12],1-CF*h*Jf[13][13],-CF*h*Jf[13][14],-CF*h*Jf[13][15]],
                             [-CF*h*Jf[14][0],-CF*h*Jf[14][1],-CF*h*Jf[14][2],-CF*h*Jf[14][3],-CF*h*Jf[14][4],-CF*h*Jf[14][5],-CF*h*Jf[14][6],-CF*h*Jf[14][7],-CF*h*Jf[14][8],-CF*h*Jf[14][9],-CF*h*Jf[14][10],-CF*h*Jf[14][11],-CF*h*Jf[14][12],-CF*h*Jf[14][13],1-CF*h*Jf[14][14],-CF*h*Jf[14][15]],
                             [-CF*h*Jf[15][0],-CF*h*Jf[15][1],-CF*h*Jf[15][2],-CF*h*Jf[15][3],-CF*h*Jf[15][4],-CF*h*Jf[15][5],-CF*h*Jf[15][6],-CF*h*Jf[15][7],-CF*h*Jf[15][8],-CF*h*Jf[15][9],-CF*h*Jf[15][10],-CF*h*Jf[15][11],-CF*h*Jf[15][12],-CF*h*Jf[15][13],-CF*h*Jf[15][14],1-CF*h*Jf[15][15]]])
        
        # initial condition for Newthon's method
        x0=[y[k][i+1] for k in range(int(len(y0)))]
        #x0=[y[0][i],y[1][i],y[2][i],y[3][i],y[4][i],y[5][i],y[6][i],y[7][i],y[8][i],y[9][i],y[10][i],y[11][i],y[12][i],y[13][i],y[14][i],y[15][i]]
        root=newton(g,Jg,x0,1e-10,10)
        # root[1] = bolean flag to make sure that the newthon was successful.
        if root[1]:
            for k in range(int(len(y0))):
                y[k][i+2] = root[0][k]
        else:
            print('No solution found')
    return y,t