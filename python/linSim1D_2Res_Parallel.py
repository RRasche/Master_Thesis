import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider
from matplotlib import animation, rc
import matplotlib 

from scipy.integrate import solve_ivp
from multiprocessing import Pool, Array

matplotlib.rcParams["figure.dpi"] = 200

#dim_x = 400
dim_y = 1
dim_z = 1
bound = 0

def sin(x, t, w = 1/50):
    w *= 2 * np.pi
    return np.sin(w/4 * x - w * t)

def gauss(x, t):
    return np.exp(-(x + (t-60))**2/400)


def get_B():
    Q = np.array([object, object, object])
    Q[0] = np.asarray([[ 0,  0,  0],
                      [ 0,  0, -1],
                      [ 0,  1,  0]])

    Q[1] = np.asarray([[ 0,  0,  1],
                      [ 0,  0,  0],
                      [-1,  0,  0]])

    Q[2] = np.asarray([[ 0, -1,  0],
                      [ 1,  0,  0],
                      [ 0,  0,  0]])

    zero3 = np.asarray(np.zeros((3, 3)))
    B = np.asarray([object, object, object])

    for i in range(0, 3):
        B[i] = np.block([[zero3, Q[i]], [-Q[i], zero3]])

    return B

def get_R():
    R = np.array([object, object, object])
    s = np.sqrt(2)
    R[0] = np.asarray([[s,  0,  0,  0,  0,  0],
                       [0,  1,  0,  0,  0, -1],
                       [0,  0,  1,  0,  1,  0],
                       [0,  0,  0,  s,  0,  0],
                       [0,  1,  0,  0,  0,  1],
                       [0,  0,  1,  0,  -1,  0]]) / s

    R[1] = np.asarray([[ 0,  s,  0,  0,  0,  0],
                       [-1,  1,  0,  0,  0, -1],
                       [ 0,  0, -1,  1,  0,  0],
                       [ 0,  0,  0,  0,  s,  0],
                       [-1,  1,  0,  0,  0,  1],
                       [ 0,  0, -1, -1,  0,  0]]) / s

    R[2] = np.asarray([[0,  0,  s,  0,  0,  0],
                       [1,  0,  0,  0, -1,  0],
                       [0,  1,  0,  1,  0,  0],
                       [0,  0,  0,  0,  0,  s],
                       [1,  0,  0,  0,  1,  0],
                       [0,  1,  0, -1,  0,  0]]) / s
    return R

def get_Stencil():
    # sPlus  = np.asarray([-1/12, 1/2, -3/2, 5/6, 1/4, 0, 0]) * 2
    # sMinus = np.asarray([0, 0, -1/4, -5/6, 3/2, -1/2, 1/12]) * 2
    # sPlus  = np.asarray([-1, 1, 0]) * 3
    # sMinus = np.asarray([0, -1, 1]) * 3


    #sPlus  = np.asarray([0, 1/20, -1/2, -1/3, 1.0, -1/4, 1/30]) * 3
    #sMinus = np.asarray([-1/30, 1/4, -1.0, 1/3, 1/2, -1/20, 0]) * 3

    sMinus = np.asarray([-1/12, 1/2, -3/2, 5/6, 1/4, 0, 0]) * 4
    sPlus  = np.asarray([0, 0, -1/4, -5/6, 3/2, -1/2, 1/12]) * 4
    # sMinus = np.asarray([-1, 1, 0]) * 1
    # sPlus  = np.asarray([0, -1, 1]) * 1



    return sPlus, sMinus

def derivative_f(y, R, sPlus, sMinus, dim_x):

    def getSvloc(sp, sm, num):
        return np.diag([sp[num], sp[num], sp[num], sm[num], sm[num], sm[num]])

    df = np.zeros_like(y)
    sLen = len(sPlus)//2
    sPs1 = np.asarray([0, 0, -16/60, -45/60, 80/60, -20/60, 1/60])*4            # -1, 0, 1, 2, 4
    sPs2 = np.asarray([0, 0, -75/240, -128/240, 225/240, - 25/240, 3/240])*4    # -1, 0, 1, 3, 5
    sPs3 = np.asarray([0, 0, -192/210, +35/210, 1, -63/210, 10/210])*4          # -0.5, 0, 1, 2, 3

    sMs1 = np.asarray([-8/30, 45/30, -120/30, 80/30, 3/30, 0, 0])*4             # -1.5, -1, -0.5, 0, 1
    sMs2 = np.asarray([-15/30, 64/30, -90/30, 35/30, 6/30, 0, 0])*4             # -2, -1.5, -1, 0, 1
    sMs3 = np.asarray([-64/210, 175/210, -350/210, 189/210, 50/210, 0, 0])*4    # -2.5, -2, -1, 0, 1
    
    sPs1b = np.asarray([0, 0, -50/210, -189/210, 350/210, -175/210, 64/210])*4 		# -1, 0, 1, 2, 2.5
    sPs2b = np.asarray([0, 0, -6/30, -35/30, 90/30, -64/30, 15/30])*4                 # -1, 0, 1, 1.5, 2
    sPs3b = np.asarray([0, 0, -3/60, -80/60, 120/60, -45/60, 8/60 ])*4 	          	# -1, 0, 0.5, 1, 1.5

    sMs1b = np.asarray([-10/420, 63/420, -210/420, -35/420, 192/420, 0, 0])*4       	#-6, -4, -2, 0, 1
    sMs2b = np.asarray([-3/240, 25/240, -225/240, 128/240, 75/240, 0, 0])*4           	#-5, -3, -1, 0, 1
    sMs3b = np.asarray([-1/60, 20/60, -80/60, 45/60, 16/60, 0, 0])*4 	                	#-4, -2, -1, 0, 1
    
    n2 = dim_x//3

    for j in range(0, dim_x + 2*bound): #dim_x + 2*bound
        for i in range(len(sPlus)):


            Sv = np.diag([sPlus[i], sPlus[i], sPlus[i], sMinus[i], sMinus[i], sMinus[i]])
            if j == (n2 - 2):
                Sv = getSvloc(sPs1, sMinus, i)
            if j == (n2 - 1):
                Sv = getSvloc(sPs2, sMinus, i)
            if j == (n2):
                Sv = getSvloc(sPs3, sMs1, i)
            if j == (n2 + 1):
                Sv = getSvloc(sPlus, sMs2, i)
            if j == (n2 + 2):
                Sv = getSvloc(sPlus, sMs3, i)
            
            
                
            if j == (2*n2 - 2):
                Sv = getSvloc(sPs1b, sMinus, i)
            if j == (2*n2 - 1):
                Sv = getSvloc(sPs2b, sMinus, i)
            if j == (2*n2):
                Sv = getSvloc(sPs3b, sMs1b, i)
            if j == (2*n2 + 1):
                Sv = getSvloc(sPlus, sMs2b, i)
            if j == (2*n2 + 2):
                Sv = getSvloc(sPlus, sMs3b, i)
            
            if j >= n2 and j < 2*n2:# and j <= 250:
                Sv = Sv / 2
            
            # if j == 198:
            #     Sv = getSvloc(sPs1, sMinus, i)
            # if j == 199:
            #     Sv = getSvloc(sPs2, sMinus, i)
            # if j == 200:
            #     Sv = getSvloc(sPs3, sMs1, i)
            # if j == 201:
            #     Sv = getSvloc(sPlus, sMs2, i)
            # if j == 202:
            #     Sv = getSvloc(sPlus, sMs3, i)
            
            
                
            # if j == 398:
            #     Sv = getSvloc(sPs1b, sMinus, i)
            # if j == 399:
            #     Sv = getSvloc(sPs2b, sMinus, i)
            # if j == 400:
            #     Sv = getSvloc(sPs3b, sMs1b, i)
            # if j == 401:
            #     Sv = getSvloc(sPlus, sMs2b, i)
            # if j == 402:
            #     Sv = getSvloc(sPlus, sMs3b, i)
                
            # if j >= 200 and j < 400:# and j <= 250:
            #     Sv = Sv / 2

            temp = y[0, 0, (j - sLen + i) % (dim_x + 2 * bound), :]
            temp = np.matmul(R[0], temp)
            temp = np.matmul(Sv, temp)
            
            df[0, 0, j, :] += temp
        df[0, 0, j, :] =  np.matmul(R[0].T, df[0, 0, j, :])

    return df


def dfdt(t, y, B, R, sPlus, sMinus, dim_x):
    y = y.reshape(dim_z, dim_y, dim_x + 2 * bound, 6)
    #for i in range(3):
    #x = np.arange(-2, 1, 1)
    #y[0, 0, 198:201, 2] = gauss(x, 2 * t)#sin(x, t)
    #y[0, 0, 198:201, 4] = gauss(x, 2 * t)#sin(x, t)

    # if(t<100):
    #     y[0, 0, 0:3, 2] = sin(x, t)  #gauss(-x, t)
    #     y[0, 0, 0:3, 4] = -sin(x, t)  #gauss(-x, t)
    # else:
    #     y[0, 0, 0:3, 2] = 0
    #     y[0, 0, 0:3, 4] = 0

    #y[0, 0, 100, 5] = 1.0/np.sqrt(2) * gauss(t)

    df = derivative_f(y, R, sPlus, sMinus, dim_x)
    dtf = np.zeros_like(df)
    for j in range(0, dim_x + bound):
        dtf[0, 0, j, :] = np.matmul(B[0], df[0, 0, j, :])

    #dtf[0, 0, 3:, 2] = dtf[0, 0, 3:, 2]/9
    #dtf[0, 0, :, 4] = dtf[0, 0, :, 4]

    return dtf.flatten()


def runSim(i):
    Ezs = np.ctypeslib.as_array(Ezs_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
    Hys = np.ctypeslib.as_array(Hys_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
    resolution = resolutions[i]
    
    simDim_x = int(resolution * 3 * 2) # two full oscillations fit onto the grid (3->avg resolution, 2->#oscillations)
    n2 = simDim_x//3
        
    f = np.zeros((dim_z, dim_y, simDim_x + 2 * bound, 6))
    f[0, 0, :n2, 2] = sin(np.arange(0,n2), 0, 1/resolution)
    f[0, 0, :n2, 4] = -sin(np.arange(0,n2), 0, 1/resolution)
    f[0, 0, n2:2*n2+1, 2] = sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[0, 0, n2:2*n2+1, 4] = -sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[0, 0, 2*n2+1:, 2] = sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)
    f[0, 0, 2*n2+1:, 4] = -sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)
    
    
    f = f.flatten()
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    out = solve_ivp(dfdt, (0, 6*resolution+1), f.flatten(), t_eval = t_eval,  args=(B, R, sPlus, sMinus, simDim_x), rtol=1e-12, atol=1e-12)
    
    y = out.y
    y= y.reshape((simDim_x + 2 * bound, 6, -1))
    
    Ez = y[:, 2, :]
    Hy = y[:, 4, :]
    Ezs[i, :simDim_x, :int(np.round((6*resolution+1)*10)) + 1] = Ez       
    Hys[i, :simDim_x, :int(np.round((6*resolution+1)*10)) + 1] = Hy  
    print(str(i) + ' / ' + str(diff_res))


diff_res =  38*2 + 1
dim_x = 40 * 3 * 2 

resolutions = np.linspace(2, 40, diff_res)
maxtRes = (int(np.round(6*resolutions[-1]))+1)*10 + 1

B = get_B()
R = get_R()

sPlus, sMinus = get_Stencil()

Ezs_Ctypes = Array('d', diff_res * (dim_x + 2*bound) * maxtRes, lock=False)
Hys_Ctypes = Array('d', diff_res * (dim_x + 2*bound) * maxtRes, lock=False)

with Pool(processes=8) as pool:
    pool.map(runSim, range(diff_res))

Ezs = np.ctypeslib.as_array(Ezs_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
Hys = np.ctypeslib.as_array(Hys_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
np.save('Ezdata.npy', Ezs)
np.save("Hydata.npy", Hys)
